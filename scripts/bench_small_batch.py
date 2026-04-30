"""
Small-batch combined throughput probe: RBM block Gibbs → AE decoder.

Scans batch_size = 1 … 10 and reports median wall time and throughput at each level.
Designed for --synthetic mode so no checkpoint .pt files are required on the cluster.

Usage:
    python scripts/bench_small_batch.py \
        --ae-config-path /path/to/ae_config.yaml \
        --gibbs-steps 10000

    # With a real RBM checkpoint:
    python scripts/bench_small_batch.py \
        --ae-config-path /path/to/ae_config.yaml \
        --rbm-checkpoint /path/to/rbm.pt

    # With torch.compile for realistic A100 numbers:
    python scripts/bench_small_batch.py \
        --ae-config-path /path/to/ae_config.yaml \
        --compile --tf32
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
import torch.nn as nn
from omegaconf import OmegaConf
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.layers import reduce as energy_reduce
from model.rbm.rbm_two_partite import RBM_TwoPartite


# ── ConvTranspose3d → Linear patch (same as benchmark_sampling.py) ────────────

class _ConvT3d1x1AsLinear(nn.Module):
    def __init__(self, ct: nn.ConvTranspose3d):
        super().__init__()
        c_in, c_out, kd, kh, kw = ct.weight.shape
        self._out_shape = (c_out, kd, kh, kw)
        out_feats = c_out * kd * kh * kw
        w = ct.weight.detach().view(c_in, out_feats).T.contiguous()
        self.linear = nn.Linear(c_in, out_feats, bias=False)
        self.linear.weight = nn.Parameter(w, requires_grad=False)
        if ct.bias is not None:
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
    try:
        first_sub = model.decoder.subdecoders[0]
        ct = first_sub.layer1_1.conv
        if not isinstance(ct, nn.ConvTranspose3d):
            print("  patch: layer1_1.conv is not ConvTranspose3d — skipping")
            return
        first_sub.layer1_1.conv = _ConvT3d1x1AsLinear(ct)
        print(f"  Patched ConvTranspose3d({ct.in_channels}, {ct.out_channels}, 3³) → Linear GEMM")
    except AttributeError as e:
        print(f"  patch: {e} — skipping")


# ── Timing helpers ─────────────────────────────────────────────────────────────

def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_runs(fn, repeats=5):
    """Run fn() `repeats` times with GPU sync; return (median_s, min_s, max_s)."""
    times = []
    for _ in range(repeats):
        _gpu_sync()
        t0 = time.perf_counter()
        fn()
        _gpu_sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


# ── Synthetic setup ────────────────────────────────────────────────────────────

def _make_synthetic_binning(path, relevant_layers=(0, 1, 2, 3, 12), n_phi=14, n_r=24):
    import h5py
    n_voxels = n_r * n_phi
    with h5py.File(path, 'w') as f:
        for layer in relevant_layers:
            f[f"layer_binsize_alpha_{layer}"]  = np.full(n_voxels, 2 * np.pi / n_phi)
            f[f"layer_binstart_alpha_{layer}"] = np.tile(
                np.linspace(0, 2 * np.pi, n_phi, endpoint=False), n_r
            )
            f[f"layer_binsize_radius_{layer}"] = np.full(n_voxels, 0.1)
            f[f"layer_binstart_radius_{layer}"] = np.repeat(
                np.linspace(0.1, 0.1 * (n_r + 1), n_r, endpoint=False), n_phi
            )


def setup_synthetic(args):
    from scripts.run import setup_model as setup_model_ae

    print(f"Loading AE architecture from {args.ae_config_path}")
    ae_config = OmegaConf.load(args.ae_config_path)
    ae_config.gpu_list = [0]
    ae_config.load_state = True
    ae_config.skip_data_loading = True

    synthetic_binning = os.path.join(script_dir, "_synthetic_binning.h5")
    if not os.path.isfile(synthetic_binning):
        _make_synthetic_binning(synthetic_binning)
        print(f"  Created synthetic binning: {synthetic_binning}")
    ae_config.data.binning_path = synthetic_binning

    ae_engine = setup_model_ae(ae_config)
    print("  AE instantiated with random weights.")

    _ar_latent = ae_config.rbm.latent_nodes_per_p * 3 + ae_config.model.cond_p_size
    dummy_data = torch.zeros(1, _ar_latent)
    rbm = RBM_TwoPartite(ae_config, data=dummy_data)

    rbm_ckpt = args.rbm_checkpoint or getattr(
        OmegaConf.load(os.path.join(project_root, "config/dwave/dwave.yaml")),
        "rbm_checkpoint", None,
    )
    if rbm_ckpt and os.path.isfile(rbm_ckpt):
        epoch = rbm.load_checkpoint(rbm_ckpt, epoch=None)
        print(f"  RBM loaded from checkpoint (epoch {epoch}).")
    else:
        print("  RBM using random weights.")

    n_clamped = ae_config.model.cond_p_size
    n_hlf = 5
    cond_vec_1    = torch.randint(0, 2, (1, n_clamped), dtype=torch.float32)
    incidence_e_1 = torch.tensor([[50_000.0]])
    u_samples_1   = torch.rand(1, n_hlf)
    _, x0_1 = energy_reduce(torch.zeros(1, 5, dtype=torch.float32), incidence_e_1)

    return ae_engine, rbm, ae_config, cond_vec_1.cpu(), incidence_e_1.cpu(), u_samples_1.cpu(), x0_1.cpu()


# ── Combined bench: batch_size 1 … max_batch ──────────────────────────────────

def bench_combined_small(rbm, ae_engine, cond_vec_1, n_clamped, x0_1, u_samples_1,
                         incidence_e_1, gibbs_steps, max_batch, repeats, use_bf16):
    device = rbm.device
    print(f"\n=== Combined RBM → decoder: batch 1–{max_batch} "
          f"(gibbs_steps={gibbs_steps}, bf16={use_bf16}) ===")

    def _make_inputs(b):
        return (
            cond_vec_1.repeat(b, 1).to(device),
            x0_1.repeat(b, 1),
            u_samples_1.repeat(b, 1),
            incidence_e_1.repeat(b, 1),
        )

    def _run(cb, x0b, ub, eb, b):
        with torch.no_grad():
            vs = rbm.sample_v_given_v_clamped(cb, n_clamped, gibbs_steps=gibbs_steps, beta=1.0)
            if use_bf16:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    ae_engine.generate_showers_from_rbm(vs, x0b, ub, eb, batch_size=b)
            else:
                ae_engine.generate_showers_from_rbm(vs, x0b, ub, eb, batch_size=b)

    # warm-up at batch=1
    print("Warming up (batch=1)...")
    cb, x0b, ub, eb = _make_inputs(1)
    _run(cb, x0b, ub, eb, 1)
    del cb, x0b, ub, eb

    rows = []
    header = f"{'batch':>6}  {'median (s)':>12}  {'min (s)':>10}  {'max (s)':>10}  {'µs/sample':>12}  {'samples/s':>12}"
    print(header)
    print("-" * len(header))

    for b in range(1, max_batch + 1):
        cb, x0b, ub, eb = _make_inputs(b)
        # one untimed warm-up per batch size (catches JIT retracing at new shapes)
        _run(cb, x0b, ub, eb, b)

        med, mn, mx = timed_runs(lambda cb=cb, x=x0b, u=ub, e=eb, b=b: _run(cb, x, u, e, b),
                                 repeats=repeats)
        us_per = med / b * 1e6
        tp = b / med
        print(f"  {b:4d}  {med:12.6f}  {mn:10.6f}  {mx:10.6f}  {us_per:12.1f}  {tp:12.1f}")
        rows.append({
            "batch_size": b,
            "wall_s_median": round(med, 6),
            "wall_s_min": round(mn, 6),
            "wall_s_max": round(mx, 6),
            "per_sample_us": round(us_per, 2),
            "samples_per_sec": round(tp, 2),
        })
        del cb, x0b, ub, eb

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    torch.backends.cudnn.benchmark = True
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled.")

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../config"):
        ae_engine, rbm, ae_config, cond_vec_1, incidence_e_1, u_samples_1, x0_1 = \
            setup_synthetic(args)

    if args.patch_decoder:
        print("Patching decoder first kernel...")
        patch_decoder_first_kernel(ae_engine.model)

    if args.compile:
        print(f"Compiling AE with torch.compile(mode='{args.compile_mode}')...")
        ae_engine.model = torch.compile(ae_engine.model, mode=args.compile_mode)

    n_clamped = ae_engine._config.model.cond_p_size

    rows = bench_combined_small(
        rbm, ae_engine, cond_vec_1, n_clamped, x0_1, u_samples_1, incidence_e_1,
        gibbs_steps=args.gibbs_steps,
        max_batch=args.max_batch,
        repeats=args.repeats,
        use_bf16=args.bf16,
    )

    output = {
        "timestamp": datetime.now().isoformat(),
        "device": str(ae_engine.device),
        "options": {
            "gibbs_steps": args.gibbs_steps,
            "max_batch": args.max_batch,
            "repeats": args.repeats,
            "patch_decoder": args.patch_decoder,
            "compile": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "tf32": args.tf32,
            "bf16": args.bf16,
        },
        "results": rows,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(script_dir, f"bench_small_batch_{ts}.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Small-batch combined RBM+decoder throughput probe (batch 1–N)."
    )
    parser.add_argument("--ae-config-path", required=True,
                        help="AE architecture config YAML (copy from training cluster; .pt not needed)")
    parser.add_argument("--rbm-checkpoint", default=None,
                        help="Override RBM checkpoint path (optional)")
    parser.add_argument("--gibbs-steps", type=int, default=10_000,
                        help="Block Gibbs steps for the RBM (default: 10000)")
    parser.add_argument("--max-batch", type=int, default=10,
                        help="Largest batch size to probe (default: 10)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Timed repetitions per batch size (default: 5)")
    parser.add_argument("--patch-decoder", action="store_true",
                        help="Replace 1×1×1 ConvTranspose3d with Linear GEMM")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile() the AE model before benchmarking")
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 on Ampere+ GPUs")
    parser.add_argument("--bf16", action="store_true",
                        help="AE decode in BF16 autocast")
    args = parser.parse_args()
    main(args)
