#!/bin/bash
#SBATCH --job-name=caloquvae_bench
#SBATCH --account=def-tafirout
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=%x_%j.out

# ── Environment ────────────────────────────────────────────────────────────────
module load StdEnv/2023 python/3.11 cuda/12.2 cudnn/8.9.5.253

VENV="$HOME/.venvs/caloquvae"
source "$VENV/bin/activate"

cd "$HOME/CaloQuVAE"

# ── Run (synthetic mode — no AE .pt weights needed) ───────────────────────────
# Adjust --ae-config-path to wherever you copied the architecture YAML.
python3 scripts/benchmark_sampling.py \
    --synthetic \
    --ae-config-path "$SCRATCH/ae_layers_no_hits_best_ema_epoch33_config.yaml" \
    --skip-qpu \
    --tf32 \
    --decoder-start-chunk 256 \
    --decoder-max-chunk 4096 \
    --gibbs-steps 1000 \
    --repeats 5
