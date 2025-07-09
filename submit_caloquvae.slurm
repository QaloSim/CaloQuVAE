#!/bin/bash
#SBATCH --account=def-tafirout              # PI's account
#SBATCH --job-name=caloquvae-test           # Descriptive job name
#SBATCH --output=logs/%x-%j.out            # Logs go to logs/jobname-jobid.out
#SBATCH --time=02:00:00                    # 2 hours of wall time
#SBATCH --cpus-per-task=4                  # Number of CPU cores
#SBATCH --mem=32G                          # Amount of system RAM
#SBATCH --gpus-per-node=a100:1            # Request 1 full A100
#SBATCH --ntasks=1                         # One process/task

# Load container support
module load apptainer

# Optional: create logs folder if it doesn't exist
mkdir -p logs

# Define container path
CONTAINER=/home/lzhu/scratch/container_qml_v3.5.5.0.sif

# Define bind mounts (scratch and project are typically needed)
BIND_PATHS="/home/lzhu/scratch,/home/lzhu/projects"

# Change to repo root (adjust if needed)
cd /home/lzhu/projects/def-tafirout/lzhu/CaloQuVAE

# Run script inside container
apptainer exec --nv -B $BIND_PATHS \
    --env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    --env DWAVE_API_TOKEN= [DWAVE TOKEN] \
    $CONTAINER python scripts/runCC.py hydra.run.dir=. hydra.output_subdir=null