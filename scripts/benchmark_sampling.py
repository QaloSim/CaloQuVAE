"""
Wall-clock benchmark for the CaloQuVAE generation pipeline.

Bench A: RBM block Gibbs on GPU — throughput-optimal batch size, configurable BGS steps.
Bench B: QPU single anneal/readout — timing from D-Wave response.info['timing'].
Bench C: AE decoder throughput — batch-size sweep.

Starting point: conditioning vector from Transfusion (Transfusion overhead excluded).

Usage:
    python scripts/benchmark_sampling.py --skip-qpu          # Bench A + C only
    python scripts/benchmark_sampling.py --gibbs-steps 1000  # quick linearity check
    python scripts/benchmark_sampling.py                      # full run (one QPU call)
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

import numpy as np
import torch
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch.nn as nn

from data.layers import reduce as energy_reduce
from model.rbm.rbm_two_partite import RBM_TwoPartite
# dwave / networkx imports are deferred to the code paths that need them
# so --synthetic --skip-qpu runs with no dwave packages installed.


# ── ConvTranspose3d → Linear patch ─────────────────────────────────────────────

class _ConvT3d1x1AsLinear(nn.Module):
    """Inference-only replacement for ConvTranspose3d when input spatial dims are 1×1×1.

    ConvTranspose3d on a 1×1×1 input is algebraically identical to a matrix multiply
    (output[b,c_out,d,h,w] = sum_c_in weight[c_in,c_out,d,h,w] * input[b,c_in,0,0,0]),
    but cuDNN routes it through a slow scatter kernel rather than cuBLAS GEMM.
    This module copies the weights and exposes a Linear forward pass instead.
    """
    def __init__(self, ct: nn.ConvTranspose3d):
        super().__init__()
        c_in, c_out, kd, kh, kw = ct.weight.shape
        self._out_shape = (c_out, kd, kh, kw)
        out_feats = c_out * kd * kh * kw
        # Linear weight convention: (out_feats, c_in) = ct.weight.view(c_in, out_feats).T
        w = ct.weight.detach().view(c_in, out_feats).T.contiguous()
        self.linear = nn.Linear(c_in, out_feats, bias=False)
        self.linear.weight = nn.Parameter(w, requires_grad=False)
        if ct.bias is not None:
            # ConvTranspose3d bias[c_out] is added to every spatial output position of that channel
            b_flat = (ct.bias.detach()
                      .unsqueeze(-1).expand(c_out, kd * kh * kw)
                      .reshape(out_feats).contiguous())
            self.register_buffer('_b', b_flat)
        else:
            self._b = None

    def forward(self, x):
        out = self.linear(x.flatten(1))
        if self._b is not None:
            out = out + self._b
        return out.view(x.shape[0], *self._out_shape)


def patch_decoder_first_kernel(model):
    """Replace the 1×1×1-input ConvTranspose3d with a Linear GEMM (same weights, faster path).
    Must be called BEFORE torch.compile() so inductor traces the Linear instead of the conv."""
    try:
        first_sub = model.decoder.subdecoders[0]
        ct = first_sub.layer1_1.conv
        if not isinstance(ct, nn.ConvTranspose3d):
            print("  patch: layer1_1.conv is not ConvTranspose3d — skipping")
            return
        first_sub.layer1_1.conv = _ConvT3d1x1AsLinear(ct)
        print(f"  Patched layer1_1: ConvTranspose3d({ct.in_channels}, {ct.out_channels}, 3³)"
              f" → Linear GEMM")
    except AttributeError as e:
        print(f"  patch: {e} — skipping")


# ── Profiler ───────────────────────────────────────────────────────────────────

def profile_decoder(ae_engine, n_vis, x0_1, u_samples_1, incidence_e_1,
                    batch_size, use_bf16, n_warmup=3, n_active=3):
    """Run torch.profiler on one decode call and print the top CUDA ops by self-time."""
    from torch.profiler import profile, ProfilerActivity, schedule

    vs  = torch.zeros(batch_size, n_vis, dtype=torch.float32)
    x0b = x0_1.repeat(batch_size, 1)
    ub  = u_samples_1.repeat(batch_size, 1)
    eb  = incidence_e_1.repeat(batch_size, 1)

    def _call():
        if use_bf16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ae_engine.generate_showers_from_rbm(vs, x0b, ub, eb, batch_size=batch_size)
        else:
            ae_engine.generate_showers_from_rbm(vs, x0b, ub, eb, batch_size=batch_size)

    print(f"\n=== Profiling decoder at batch={batch_size} (bf16={use_bf16}) ===")
    print(f"Warming up {n_warmup}×...")
    for _ in range(n_warmup):
        with torch.no_grad():
            _call()

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        schedule=schedule(wait=0, warmup=0, active=n_active),
    ) as prof:
        for _ in range(n_active):
            with torch.no_grad():
                _call()
            prof.step()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


# ── Timing helpers ─────────────────────────────────────────────────────────────

def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_runs(fn, repeats=3):
    """Run fn() `repeats` times with GPU sync; return (median_s, min_s, max_s)."""
    times = []
    for _ in range(repeats):
        _gpu_sync()
        t0 = time.perf_counter()
        fn()
        _gpu_sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def _is_oom(exc):
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()



# ── Bench A: RBM block Gibbs on GPU ────────────────────────────────────────────

def bench_rbm_gpu(rbm, cond_vec_1, n_clamped, gibbs_steps, batch_size, repeats):
    """
    Times RBM block Gibbs at a fixed batch_size (typically the decoder-optimal batch).
    cond_vec_1: shape [1, n_clamped].
    Returns (result_dict, v_samples).
    """
    device = rbm.device
    print(f"\n=== Bench A: RBM block Gibbs (GPU) — batch={batch_size}, steps={gibbs_steps} ===")

    cond_batch = cond_vec_1.repeat(batch_size, 1).to(device)

    print("Warm-up...")
    with torch.no_grad():
        rbm.sample_v_given_v_clamped(cond_batch, n_clamped, gibbs_steps=gibbs_steps, beta=1.0)

    v_final = [None]

    def run():
        with torch.no_grad():
            v_final[0] = rbm.sample_v_given_v_clamped(
                cond_batch, n_clamped, gibbs_steps=gibbs_steps, beta=1.0
            )

    print(f"Timing {repeats}×...")
    med, mn, mx = timed_runs(run, repeats=repeats)

    per_sample_us = (med / batch_size) * 1e6
    per_step_ms   = (med / gibbs_steps) * 1e3
    samples_per_sec = batch_size / med

    row = {
        "batch": batch_size,
        "gibbs_steps": gibbs_steps,
        "n_clamped": n_clamped,
        "wall_s_median": round(med, 4),
        "wall_s_min": round(mn, 4),
        "wall_s_max": round(mx, 4),
        "per_sample_us": round(per_sample_us, 2),
        "per_step_ms": round(per_step_ms, 4),
        "samples_per_sec": round(samples_per_sec, 1),
    }
    print(f"  Wall: {med:.3f}s  (min={mn:.3f}  max={mx:.3f})")
    print(f"  Throughput: {samples_per_sec:.0f} samples/s  |  {per_sample_us:.1f} µs/sample  |  {per_step_ms:.4f} ms/step")

    return row, v_final[0]


# ── Bench B: QPU single anneal/readout ────────────────────────────────────────

def bench_qpu(rbm, raw_sampler, cond_sets, left_chains, right_chains, hidden_side, cond_vec_1):
    """
    Single num_reads=1 QPU call. Extracts response.info['timing'] (values in µs).
    """
    from utils.dwave.workflows import sample_expanded_flux_arbitrary
    from utils.dwave.graphs import get_orbit_mappings

    print("\n=== Bench B: QPU single anneal/readout ===")

    if hidden_side == 'right':
        n_vis_orbit, n_hid_orbit = len(left_chains), len(right_chains)
    else:
        n_vis_orbit, n_hid_orbit = len(right_chains), len(left_chains)

    vis_mapping, hid_mapping = get_orbit_mappings(0, n_vis_orbit, n_hid_orbit)

    print("Submitting num_reads=1 to QPU (one anneal, minimal QPU budget)...")
    analysis = sample_expanded_flux_arbitrary(
        rbm=rbm,
        raw_sampler=raw_sampler,
        conditioning_sets=cond_sets,
        left_chains=left_chains,
        right_chains=right_chains,
        binary_patterns_batch=cond_vec_1,  # shape [1, n_clamped] → num_reads=1
        hidden_side=hidden_side,
        beta=1.0,
        chain_strength=2.0,
        use_srt=True,
        logical_srt=True,
        flux_drift_compensation=True,
        vis_mapping=vis_mapping,
        hid_mapping=hid_mapping,
        perm_seed=0,
        source="bench_b",
        save_dir=None,
    )

    raw_timing = analysis.physical_response.info.get('timing', {})

    # Canonical keys; D-Wave reports these in µs
    keys = [
        'qpu_anneal_time_per_sample',
        'qpu_readout_time_per_sample',
        'qpu_sampling_time',
        'qpu_programming_time',
        'qpu_access_time',
        'qpu_access_overhead_time',
    ]
    timing_us = {k: raw_timing[k] for k in keys if k in raw_timing}

    anneal = timing_us.get('qpu_anneal_time_per_sample', float('nan'))
    readout = timing_us.get('qpu_readout_time_per_sample', float('nan'))
    prog = timing_us.get('qpu_programming_time', float('nan'))

    print(f"  Anneal/sample:  {anneal:.1f} µs")
    print(f"  Readout/sample: {readout:.1f} µs")
    print(f"  Anneal+Readout: {anneal + readout:.1f} µs  (GPU-comparable cost)")
    print(f"  Programming:    {prog:.1f} µs  (amortised across reads in production)")
    if raw_timing:
        extra = {k: v for k, v in raw_timing.items() if k not in keys}
        if extra:
            print(f"  Other keys: {extra}")

    return {"timing_us": timing_us, "all_timing_keys": list(raw_timing.keys())}


# ── Bench C: AE decoder throughput ────────────────────────────────────────────

def bench_decoder(ae_engine, n_vis, x0_1, u_samples_1, incidence_e_1, start_chunk, repeats,
                  max_chunk=8192, use_bf16=False):
    """
    Sweeps batch_size = total_samples together from start_chunk upward (doubling), so each
    level measures a single GPU forward pass of that size.  cuDNN often finds faster kernel
    algorithms at larger batch sizes, so we must sweep past the plateau that appears at
    small batches.  Stops at OOM or max_chunk.  Uses `repeats` timed passes at every sweep
    level (not just the final one) so per-level medians are stable.
    Returns (result_dict, optimal_chunk_size).
    """
    print("\n=== Bench C: AE decoder throughput ===")
    print(f"Single-pass sweep (start={start_chunk}, max={max_chunk}, {repeats} repeats/level)...")

    chunk = start_chunk
    best_chunk = start_chunk
    best_tp = 0.0
    sweep = []

    def _make_inputs(b):
        vs  = torch.zeros(b, n_vis, dtype=torch.float32)
        x0b = x0_1.repeat(b, 1)
        ub  = u_samples_1.repeat(b, 1)
        eb  = incidence_e_1.repeat(b, 1)
        return vs, x0b, ub, eb

    # Warm-up at start_chunk
    def _decode(vs, x0b, ub, eb, c, use_bf16):
        if use_bf16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ae_engine.generate_showers_from_rbm(vs, x0b, ub, eb, batch_size=c)
        else:
            ae_engine.generate_showers_from_rbm(vs, x0b, ub, eb, batch_size=c)

    print(f"Warming up at start_chunk={start_chunk}...")
    vs, x0b, ub, eb = _make_inputs(start_chunk)
    with torch.no_grad():
        _decode(vs, x0b, ub, eb, start_chunk, use_bf16)
    del vs, x0b, ub, eb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    while chunk <= max_chunk:
        try:
            vs, x0b, ub, eb = _make_inputs(chunk)
            # One untimed warm-up per chunk: absorbs torch.compile JIT tracing for new shapes
            with torch.no_grad():
                _decode(vs, x0b, ub, eb, chunk, use_bf16)

            def run(vs=vs, x0b=x0b, ub=ub, eb=eb, c=chunk):
                with torch.no_grad():
                    _decode(vs, x0b, ub, eb, c, use_bf16)

            med, mn, mx = timed_runs(run, repeats=repeats)
            tp = chunk / med
            per_us = med / chunk * 1e6
            del vs, x0b, ub, eb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  batch {chunk}: OOM — stopping sweep")
            break

        marker = ""
        if tp > best_tp:
            best_tp = tp
            best_chunk = chunk
            marker = " ← best"

        sweep.append({
            "chunk_size": chunk,
            "throughput_samples_per_s": round(tp, 1),
            "wall_s_median": round(med, 4),
            "wall_s_min": round(mn, 4),
            "wall_s_max": round(mx, 4),
            "per_sample_us": round(per_us, 2),
        })
        print(f"  batch {chunk:8d}: {tp:>10.0f} samples/s  ({per_us:.1f} µs/sample)"
              f"  [{mn:.3f}–{mx:.3f}s]{marker}")

        chunk *= 2

    # Precise final timing at the optimal chunk (already warmed up during sweep)
    print(f"\nFinal timing {repeats}× at optimal batch={best_chunk}...")
    vs, x0b, ub, eb = _make_inputs(best_chunk)

    def final_run(vs=vs, x0b=x0b, ub=ub, eb=eb, c=best_chunk):
        with torch.no_grad():
            _decode(vs, x0b, ub, eb, c, use_bf16)

    med, mn, mx = timed_runs(final_run, repeats=repeats)
    per_us_final = med / best_chunk * 1e6
    print(f"  {med:.3f}s  ({per_us_final:.1f} µs/sample)")

    result = {
        "sweep": sweep,
        "optimum_chunk": best_chunk,
        "wall_s_median": round(med, 4),
        "wall_s_min": round(mn, 4),
        "wall_s_max": round(mx, 4),
        "per_sample_us": round(per_us_final, 2),
    }
    return result, best_chunk


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(result_a, result_b, result_c):
    print("\n" + "=" * 62)
    print("BENCHMARK SUMMARY")
    print("=" * 62)

    if result_a is not None:
        a = result_a[0]
        print(f"\n[A] RBM block Gibbs — GPU")
        print(f"    Batch (decoder-optimal): {a['batch']}")
        print(f"    BGS steps:     {a['gibbs_steps']}")
        print(f"    Wall:          {a['wall_s_median']:.3f}s  (min {a['wall_s_min']:.3f}  max {a['wall_s_max']:.3f})")
        print(f"    Throughput:    {a['samples_per_sec']:.0f} samples/s")
        print(f"    Per-sample:    {a['per_sample_us']:.1f} µs")
        print(f"    Per-step:      {a['per_step_ms']:.4f} ms/step")

    if result_b is not None:
        t = result_b.get('timing_us', {})
        anneal = t.get('qpu_anneal_time_per_sample', float('nan'))
        readout = t.get('qpu_readout_time_per_sample', float('nan'))
        prog = t.get('qpu_programming_time', float('nan'))
        print(f"\n[B] QPU — single anneal/readout")
        print(f"    Anneal/sample:  {anneal:.1f} µs")
        print(f"    Readout/sample: {readout:.1f} µs")
        print(f"    Anneal+Readout: {anneal + readout:.1f} µs  ← GPU-comparable")
        print(f"    Programming:    {prog:.1f} µs  (amortised in production)")
    else:
        print("\n[B] QPU — skipped")

    if result_c is not None:
        best_c = result_c['optimum_chunk']
        print(f"\n[C] AE decoder")
        print(f"    Optimal batch: {best_c}  ({result_c['per_sample_us']:.1f} µs/sample)")
        print(f"    Wall:          {result_c['wall_s_median']:.3f}s  (min {result_c['wall_s_min']:.3f}  max {result_c['wall_s_max']:.3f})")
        print(f"    Sweep:")
        for row in result_c['sweep']:
            marker = " ← optimal" if row['chunk_size'] == best_c else ""
            print(f"      batch={row['chunk_size']:8d}: {row['per_sample_us']:.1f} µs/sample{marker}")

    print("=" * 62)


# ── JSON serialisation helper ──────────────────────────────────────────────────

def _default_serial(obj):
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None  # NaN / Inf → null
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


# ── Synthetic setup (no checkpoint files required for AE weights) ──────────────

def _make_synthetic_binning(path, relevant_layers=(0, 1, 2, 3, 12), n_phi=14, n_r=24):
    """Write a minimal HDF5 binning file with uniform synthetic geometry.

    AtlasGeometry reads binstart/binsize arrays for radius and alpha for each layer.
    The values don't affect decoder timing — they're only used by the feature extractor
    during training.  We write uniform grids so the arrays have the right length and
    the model constructor doesn't crash.
    """
    import h5py
    n_voxels = n_r * n_phi  # 336 for standard ATLAS config
    with h5py.File(path, 'w') as f:
        for layer in relevant_layers:
            prefix = f"layer_{layer}"
            f[f"{prefix}_binsize_alpha"]  = np.full(n_voxels, 2 * np.pi / n_phi)
            f[f"{prefix}_binstart_alpha"] = np.tile(
                np.linspace(0, 2 * np.pi, n_phi, endpoint=False), n_r
            )
            f[f"{prefix}_binsize_radius"] = np.full(n_voxels, 0.1)
            f[f"{prefix}_binstart_radius"] = np.repeat(
                np.linspace(0.1, 0.1 * (n_r + 1), n_r, endpoint=False), n_phi
            )


def _setup_synthetic(args):
    """
    Build AE engine + RBM with correct architecture but random (untrained) weights.

    Requires:
      --ae-config-path  the small YAML saved alongside the AE checkpoint
                        (e.g. ae_layers_no_hits_best_ema_epoch33_config.yaml).
                        Copy this file from your training cluster; it is plain text, ~10 KB.
      --rbm-checkpoint  optional; if omitted, RBM also uses random weights.

    AE weight .pt files are NOT needed — timing is determined by architecture, not values.
    """
    from scripts.run import setup_model as setup_model_ae

    print(f"\nSynthetic mode: loading AE architecture from {args.ae_config_path}")
    ae_config = OmegaConf.load(args.ae_config_path)
    ae_config.gpu_list = [0]
    ae_config.load_state = True        # skip feature-stats init from dataset
    ae_config.skip_data_loading = True # skip DataManager construction entirely

    # AtlasGeometry is instantiated in the model __init__ and opens the binning HDF5
    # unconditionally.  Generate a tiny synthetic one so the constructor succeeds.
    synthetic_binning = os.path.join(script_dir, "_synthetic_binning.h5")
    if not os.path.isfile(synthetic_binning):
        _make_synthetic_binning(synthetic_binning)
        print(f"  Created synthetic binning file: {synthetic_binning}")
    ae_config.data.binning_path = synthetic_binning

    ae_engine = setup_model_ae(ae_config)
    print("  AE model instantiated with random weights (no .pt loaded).")

    dummy_data = torch.zeros(1, ae_config.rbm.latent_nodes_per_p)
    rbm = RBM_TwoPartite(ae_config, data=dummy_data)
    rbm_ckpt = args.rbm_checkpoint or getattr(
        OmegaConf.load(os.path.join(project_root, "config/dwave/dwave.yaml")),
        "rbm_checkpoint", None,
    )
    if rbm_ckpt and os.path.isfile(rbm_ckpt):
        loaded = rbm.load_checkpoint(rbm_ckpt, epoch=None)
        print(f"  RBM loaded from checkpoint (epoch {loaded}).")
    else:
        print("  RBM using random weights (no checkpoint found/given).")

    n_clamped = ae_engine._config.model.cond_p_size
    n_hlf = 5   # transform_dataset always returns 5 features (u0 + 4 fractions)
    cond_vec_1    = torch.randint(0, 2, (1, n_clamped), dtype=torch.float32)
    incidence_e_1 = torch.tensor([[50_000.0]])   # 50 GeV in MeV
    u_samples_1   = torch.rand(1, n_hlf)
    _, x0_1 = energy_reduce(torch.zeros(1, 5, dtype=torch.float32), incidence_e_1)
    print(f"  Synthetic cond_vec: shape={list(cond_vec_1.shape)}, "
          f"u_samples: shape={list(u_samples_1.shape)}")

    return ae_engine, rbm, ae_config, cond_vec_1, incidence_e_1, u_samples_1, x0_1


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    torch.backends.cudnn.benchmark = True
    if args.tf32:
        # TF32 uses tensor cores for matmul/conv on Ampere+ GPUs with negligible precision loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled.")

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../config"):
        if args.synthetic:
            ae_engine, rbm, ae_config, cond_vec_1, incidence_e_1, u_samples_1, x0_1 = \
                _setup_synthetic(args)
            x0_1 = x0_1.cpu()
        else:
            from scripts.dwave_samples import setup_engines
            from utils.dwave.physics import get_cond_vec
            ae_engine, tf_engine, ae_config = setup_engines()
            dwave_cfg_tmp = OmegaConf.load(os.path.join(project_root, "config/dwave/dwave.yaml"))
            dummy_data = torch.zeros(1, ae_config.rbm.latent_nodes_per_p)
            rbm = RBM_TwoPartite(ae_config, data=dummy_data)
            loaded = rbm.load_checkpoint(dwave_cfg_tmp.rbm_checkpoint, epoch=None)
            print(f"Loaded RBM checkpoint epoch {loaded}.")

            energy_mev = 50_000.0
            print(f"\nGenerating 1 conditioning vector at {energy_mev:.0f} MeV "
                  f"(Transfusion call excluded from bench)...")
            energy_tensor = torch.tensor([[energy_mev]], dtype=torch.float32)
            cond_vec_1, incidence_e_1, u_samples_1, _ = get_cond_vec(energy_tensor, ae_engine, tf_engine)
            cond_vec_1    = cond_vec_1.cpu()
            incidence_e_1 = incidence_e_1.cpu()
            u_samples_1   = u_samples_1.cpu()
            _, x0_1 = energy_reduce(torch.zeros(1, 5, dtype=torch.float32), incidence_e_1)
            x0_1 = x0_1.cpu()

    dwave_cfg = OmegaConf.load(os.path.join(project_root, "config/dwave/dwave.yaml"))
    hidden_side = dwave_cfg.sampling.hidden_side

    if args.patch_decoder:
        print("Patching AE decoder first kernel...")
        patch_decoder_first_kernel(ae_engine.model)

    if args.compile:
        print(f"Compiling AE decoder with torch.compile(mode='{args.compile_mode}')...")
        ae_engine.model = torch.compile(ae_engine.model, mode=args.compile_mode)
    n_clamped = ae_engine._config.model.cond_p_size

    # Embedding — only needed for QPU bench
    sampler = left_chains = right_chains = cond_sets = None
    if not args.skip_qpu:
        from scripts.dwave_samples import setup_embedding
        sampler, left_chains, right_chains, cond_sets, hidden_side = setup_embedding(ae_engine, dwave_cfg)

    output = {
        "timestamp": datetime.now().isoformat(),
        "device": str(ae_engine.device),
        "options": {
            "patch_decoder": args.patch_decoder,
            "compile": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "tf32": args.tf32,
            "bf16": args.bf16,
        },
        "rbm": {
            "n_vis": int(rbm.params["vbias"].shape[0]),
            "n_hid": int(rbm.params["hbias"].shape[0]),
            "n_clamped": n_clamped,
        },
        "bench_a_rbm_gpu": None,
        "bench_b_qpu": None,
        "bench_c_decoder": None,
    }

    n_vis = int(rbm.params["vbias"].shape[0])

    # ── Bench C first — determines shared chunk size for Bench A ──
    r_c, optimal_chunk = bench_decoder(
        ae_engine, n_vis, x0_1, u_samples_1, incidence_e_1,
        start_chunk=args.decoder_start_chunk,
        repeats=args.repeats,
        max_chunk=args.decoder_max_chunk,
        use_bf16=args.bf16,
    )
    output["bench_c_decoder"] = r_c

    if args.profile:
        profile_decoder(ae_engine, n_vis, x0_1, u_samples_1, incidence_e_1,
                        batch_size=optimal_chunk, use_bf16=args.bf16)

    # ── Bench A — uses the decoder-optimal chunk size as RBM batch ──
    r_a = bench_rbm_gpu(
        rbm, cond_vec_1, n_clamped,
        gibbs_steps=args.gibbs_steps,
        batch_size=optimal_chunk,
        repeats=args.repeats,
    )
    output["bench_a_rbm_gpu"] = r_a[0]

    # ── Bench B ──
    if not args.skip_qpu:
        r_b = bench_qpu(rbm, sampler, cond_sets, left_chains, right_chains, hidden_side, cond_vec_1)
        output["bench_b_qpu"] = r_b
    else:
        r_b = None
        print("\n=== Bench B: QPU (skipped via --skip-qpu) ===")

    # ── Print and save ──
    print_summary(r_a, r_b, r_c)

    outfile = os.path.join(
        script_dir,
        f"benchmark_sampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=_default_serial)
    print(f"\nResults saved → {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wall-clock benchmark: GPU RBM Gibbs vs QPU anneal vs AE decode."
    )
    parser.add_argument("--gibbs-steps", type=int, default=10_000,
                        help="BGS steps for Bench A (default: 10000)")
    parser.add_argument("--decoder-start-chunk", type=int, default=256,
                        help="Starting batch size for decoder sweep (default: 256)")
    parser.add_argument("--decoder-max-chunk", type=int, default=4096,
                        help="Maximum batch size for decoder sweep before OOM stops it (default: 131072)")
    parser.add_argument("--skip-qpu", action="store_true",
                        help="Skip Bench B (no QPU calls)")
    parser.add_argument("--patch-decoder", action="store_true",
                        help="Replace the 1×1×1-input ConvTranspose3d with a Linear GEMM (no retraining needed; apply before --compile)")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile() to the AE model before benchmarking")
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: 'default'; 'max-autotune' searches longer for faster triton kernels)")
    parser.add_argument("--profile", action="store_true",
                        help="Run torch.profiler at the optimal batch after bench_C and print top CUDA ops")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 on Ampere+ GPUs (free speedup, negligible precision change)")
    parser.add_argument("--bf16", action="store_true",
                        help="Run AE decode in BF16 autocast (~2x speedup, halves memory bandwidth)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timed repetitions per measurement (default: 3)")
    # ── Synthetic mode ──────────────────────────────────────────────────────────
    parser.add_argument("--synthetic", action="store_true",
                        help="Use random AE weights + synthetic cond vectors — "
                             "no AE .pt checkpoint needed, only the architecture config YAML "
                             "(see --ae-config-path). RBM checkpoint is loaded if accessible.")
    parser.add_argument("--ae-config-path", default=None,
                        help="Path to the AE architecture config YAML saved alongside the "
                             "checkpoint (required with --synthetic). Copy this small file "
                             "from your training cluster; the .pt weight files are not needed.")
    parser.add_argument("--rbm-checkpoint", default=None,
                        help="Override the RBM checkpoint path from dwave.yaml "
                             "(useful in --synthetic mode if the path in dwave.yaml is wrong "
                             "for this machine).")
    args = parser.parse_args()
    if args.synthetic and not args.ae_config_path:
        parser.error("--synthetic requires --ae-config-path")
    main(args)
