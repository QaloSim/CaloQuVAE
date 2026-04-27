import torch
import json
import os

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from scripts.run import setup_model as setup_model_ae
from scripts.run_transfusion import setup_model as setup_model_transfusion
from model.rbm.rbm_two_partite import RBM_TwoPartite
from utils.dwave.workflows import find_beta_single, mass_sample_dwave_single
from utils.dwave.physics import get_cond_vec
from utils.dwave.graphs import (
    run_embedding,
    select_optimal_side_exact,
    augment_cond_sets_from_visible_chains,
)


def setup_engines(ae_cfg_name="config_layers.yaml", tf_cfg_name="tfusion_config.yaml"):
    ae_cfg = compose(config_name=ae_cfg_name)
    ae_config = OmegaConf.load(ae_cfg.config_path)
    ae_config.gpu_list = ae_cfg.gpu_list
    ae_config.load_state = True
    ae_engine = setup_model_ae(ae_config)
    ae_engine._model_creator.load_state(ae_config.run_path, ae_engine.device)

    tf_cfg = compose(config_name=tf_cfg_name)
    tf_config = OmegaConf.load(tf_cfg.config_path)
    tf_config.gpu_list = ae_config.gpu_list
    tf_config.load_state = tf_config.load_state
    tf_engine = setup_model_transfusion(tf_config)
    tf_engine._model_creator.load_state(tf_config.run_path, tf_engine.device)

    return ae_engine, tf_engine, ae_config


def setup_rbm(ae_config, checkpoint_file):
    dummy_data = torch.zeros(1, ae_config.rbm.latent_nodes_per_p)
    rbm = RBM_TwoPartite(ae_config, data=dummy_data)
    loaded_epoch = rbm.load_checkpoint(checkpoint_file, epoch=None)
    print(f"Loaded RBM checkpoint epoch {loaded_epoch}.")
    return rbm


def setup_embedding(ae_engine, cfg):
    n_visible = cfg.n_vis
    n_hidden = cfg.n_hid
    print(f"RBM dimensions: n_visible={n_visible}, n_hidden={n_hidden}")

    num_clamped_bits = ae_engine._config.model.cond_p_size
    sampler, _, q_used, left_chains, right_chains = run_embedding(
        n_visible, n_hidden, cfg.solver_name
    )
    cond_sets, selected_side = select_optimal_side_exact(
        sampler, q_used, left_chains, right_chains
    )

    cond_sets = cond_sets[:num_clamped_bits]
    n_needed = num_clamped_bits - len(cond_sets)
    if n_needed > 0:
        n_vis_rbm = ae_engine._config.rbm.latent_nodes_per_p * 3
        cond_sets, left_chains, right_chains, _ = augment_cond_sets_from_visible_chains(
            cond_sets, left_chains, right_chains,
            selected_side=selected_side,
            n_extra=n_needed,
            n_visible=n_vis_rbm,
        )

    print(f"Embedding ready: {len(left_chains)} left, {len(right_chains)} right, "
          f"{len(cond_sets)} cond sets, hidden_side={selected_side}")
    return sampler, left_chains, right_chains, cond_sets, selected_side


def estimate_beta(rbm, sampler, left_chains, right_chains, cond_sets, hidden_side,
                  cond_vec, cfg, single_batch=False):
    optimal_beta, beta_hist, rbm_e_hist, qpu_e_hist = find_beta_single(
        rbm=rbm,
        raw_sampler=sampler,
        conditioning_sets=cond_sets,
        left_chains=left_chains,
        right_chains=right_chains,
        binary_patterns_batch=cond_vec,
        hidden_side=hidden_side,
        logical_srt=cfg.sampling.logical_srt,
        orbit_seed=cfg.sampling.orbit_seed,
        num_reads=cfg.beta.num_reads,
        rbm_gibbs_steps=cfg.beta.rbm_gibbs_steps,
        rbm_factor=cfg.beta.rbm_factor,
        beta_init=cfg.beta.init,
        lr=cfg.beta.lr,
        num_epochs=cfg.beta.num_epochs,
        tolerance=cfg.beta.tolerance,
        single_batch=single_batch,
        use_identity_orbit=cfg.sampling.get("use_identity_orbit", False),
        flux_drift_compensation=True,
    )
    return optimal_beta, beta_hist, rbm_e_hist, qpu_e_hist


def save_beta_info(save_dir, beta, beta_hist, rbm_e_hist, qpu_e_hist):
    info = {
        "beta_final": beta,
        "beta_hist": beta_hist,
        "rbm_e_hist": rbm_e_hist,
        "qpu_e_hist": qpu_e_hist,
    }
    path = os.path.join(save_dir, "beta_info.json")
    with open(path, "w") as f:
        json.dump(info, f)
    print(f"Saved beta info to {path}")


def run_uniform_range(cfg, ae_engine, tf_engine, rbm, sampler,
                      left_chains, right_chains, cond_sets, selected_side):
    ur = cfg.uniform_range
    print(f"\n{'='*20} Uniform Range Run ({ur.min_energy}-{ur.max_energy} MeV) {'='*20}")

    energy_tensor = torch.zeros(ur.num_samples, 1).uniform_(ur.min_energy, ur.max_energy)
    cond_vec, incidence_energy, u_samples, E_samples = get_cond_vec(
        energy_tensor, ae_engine, tf_engine
    )

    print("Estimating beta...")
    optimal_beta, beta_hist, rbm_e_hist, qpu_e_hist = estimate_beta(
        rbm, sampler, left_chains, right_chains, cond_sets, selected_side,
        cond_vec[:cfg.beta.num_reads], cfg, single_batch=True,
    )
    print(f"Optimal beta: {optimal_beta:.4f}")

    save_dir = os.path.join(cfg.sampling.output_dir, "uniform_range")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(incidence_energy, os.path.join(save_dir, "incidence_energy.pt"))
    torch.save(u_samples, os.path.join(save_dir, "u_samples.pt"))
    torch.save(E_samples, os.path.join(save_dir, "E_samples.pt"))
    save_beta_info(save_dir, optimal_beta, beta_hist, rbm_e_hist, qpu_e_hist)

    print(f"Collecting {ur.num_samples} samples (one QPU call per pattern)...")
    mass_sample_dwave_single(
        cond_vec=cond_vec,
        save_dir=save_dir,
        rbm=rbm,
        raw_sampler=sampler,
        conditioning_sets=cond_sets,
        left_chains=left_chains,
        right_chains=right_chains,
        beta=optimal_beta,
        hidden_side=selected_side,
        orbit_seed=cfg.sampling.orbit_seed,
        logical_srt=cfg.sampling.logical_srt,
        print_interval=50,
        use_identity_orbit=cfg.sampling.get("use_identity_orbit", False),
        flux_drift_compensation=True,
    )


def run_dedicated_energy(energy_mev, cfg, ae_engine, tf_engine, rbm, sampler,
                         left_chains, right_chains, cond_sets, selected_side):
    print(f"\n{'='*20} Dedicated Energy {energy_mev} MeV {'='*20}")

    n = cfg.dedicated.num_samples
    energy_tensor = torch.full((n, 1), float(energy_mev))
    cond_vec, incidence_energy, u_samples, E_samples = get_cond_vec(
        energy_tensor, ae_engine, tf_engine
    )

    print(f"Estimating beta for {energy_mev} MeV...")
    optimal_beta, beta_hist, rbm_e_hist, qpu_e_hist = estimate_beta(
        rbm, sampler, left_chains, right_chains, cond_sets, selected_side,
        cond_vec[:cfg.beta.num_reads], cfg, single_batch=True,
    )
    print(f"Optimal beta: {optimal_beta:.4f}")

    save_dir = os.path.join(cfg.sampling.output_dir, f"energy_{energy_mev}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(incidence_energy, os.path.join(save_dir, "incidence_energy.pt"))
    torch.save(u_samples, os.path.join(save_dir, "u_samples.pt"))
    torch.save(E_samples, os.path.join(save_dir, "E_samples.pt"))
    save_beta_info(save_dir, optimal_beta, beta_hist, rbm_e_hist, qpu_e_hist)

    print(f"Collecting {n} samples for {energy_mev} MeV...")
    mass_sample_dwave_single(
        cond_vec=cond_vec,
        save_dir=save_dir,
        rbm=rbm,
        raw_sampler=sampler,
        conditioning_sets=cond_sets,
        left_chains=left_chains,
        right_chains=right_chains,
        beta=optimal_beta,
        hidden_side=selected_side,
        orbit_seed=cfg.sampling.orbit_seed,
        logical_srt=cfg.sampling.logical_srt,
        print_interval=50,
        use_identity_orbit=cfg.sampling.get("use_identity_orbit", False),
        flux_drift_compensation=True,
    )


def main(cfg, ae_engine, tf_engine, ae_config):
    rbm = setup_rbm(ae_config, cfg.rbm_checkpoint)

    sampler, left_chains, right_chains, cond_sets, selected_side = setup_embedding(
        ae_engine, cfg
    )

    os.makedirs(cfg.sampling.output_dir, exist_ok=True)

    if cfg.uniform_range.enabled:
        run_uniform_range(
            cfg, ae_engine, tf_engine, rbm, sampler,
            left_chains, right_chains, cond_sets, selected_side,
        )

    if cfg.dedicated.get("enabled", True):
        for energy in cfg.dedicated.energies:
            run_dedicated_energy(
                energy, cfg, ae_engine, tf_engine, rbm, sampler,
                left_chains, right_chains, cond_sets, selected_side,
            )

    print("\nScript finished.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    dwave_cfg = OmegaConf.load(os.path.join(project_root, "config/dwave/dwave.yaml"))

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../config"):
        ae_engine, tf_engine, ae_config = setup_engines()

    main(dwave_cfg, ae_engine, tf_engine, ae_config)
