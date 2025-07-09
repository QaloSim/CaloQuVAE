#!/bin/bash
#SBATCH --account=def-tafirout              # PI's account
#SBATCH --job-name=caloqvae-test           # Descriptive job name
#SBATCH --output=logs/%x-%j.out            # Logs go to logs/jobname-jobid.out
#SBATCH --time=02:00:00                    # 2 hours of wall time
#SBATCH --cpus-per-task=4                  # Number of CPU cores
#SBATCH --mem=32G                          # Amount of system RAM
#SBATCH --gpus-per-node=a100:1            # Request 1 full A100
#SBATCH --ntasks=1                         # One process/task
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lzhu@triumf.ca


# Load container support
module load apptainer

# Add current directory to PATH
export PYTHONPATH=/home/lzhu/projects/def-tafirout/lzhu:/home/lzhu/projects/def-tafirout/lzhu/CaloQuVAE:$PYTHONPATH

# create logs folder if it doesn't exist
mkdir -p logs

# Copy the container to the compute node's fast local storage ($SLURM_TMPDIR)
echo "Copying container to node-local storage..."
cp /home/lzhu/scratch/container_qml_v3.5.5.0.sif $SLURM_TMPDIR
echo "Copy complete."

# Define the container path to be the one on the local node
CONTAINER=$SLURM_TMPDIR/container_qml_v3.5.5.0.sif

# Define bind paths 
BIND_PATHS="/home/lzhu/scratch,/home/lzhu/projects/def-tafirout/lzhu/CaloQuVAE"

# Change to repo root
cd /home/lzhu/projects/def-tafirout/lzhu/CaloQuVAE

# Export environment variables
export DWAVE_API_TOKEN="4Fq8-98e991a786ff04f4f4b6ab5466629411358ce418"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Run script inside container, using the fast local copy
apptainer exec --nv -B $BIND_PATHS \
    $CONTAINER python scripts/runCC.py hydra.run.dir=. hydra.output_subdir=null