#!/bin/bash

#SBATCH --account=def-steinman
#SBATCH --job-name=coronary
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Requesting resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=23:59:00

# Export all environment vars
#SBATCH --export=ALL


# Navigate to the script directory
cd $SLURM_SUBMIT_DIR

# Compute how many ranks to launch
NP="${SLURM_NTASKS:-1}"

module load StdEnv/2023 gcc/12.3 #openmpi/4.1.5

# Set writable directories for caching (and ensure the directories exist)
CACHE_DIR="/scratch/$USER/Torino/Coronary/scripts/.cache/"
mkdir -p "$CACHE_DIR"


# Pass env into the container (Apptainer strips the prefix)
export APPTAINERENV_DIJITSO_CACHE_DIR="$CACHE_DIR"
export APPTAINERENV_FFC_CACHE_DIR="$CACHE_DIR"
#export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
#export APPTAINERENV_OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS"
#export MPI_DIR="<PATH/TO/HOST/MPI/DIRECTORY>"

# Bind what you need; add more binds if your data/code live elsewhere
BIND_OPTS="--bind /scratch:/scratch --bind $SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR --pwd $SLURM_SUBMIT_DIR"

# Run in serial
#APPTAINERENV_PYTHONPATH=/scratch/ranbar/pyshims:$PYTHONPATH \
#apptainer exec $BIND_OPTS ~/fenics-legacy-updated2.sif oasis NSfracStep problem=coronary

# Run in parallel
APPTAINERENV_PYTHONPATH=/scratch/ranbar/pyshims:$PYTHONPATH \
apptainer exec --env HYDRA_LAUNCHER=fork $BIND_OPTS ~/fenics-legacy-updated2.sif mpirun -n "$NP" oasis NSfracStep problem=coronary