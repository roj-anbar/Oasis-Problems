#!/bin/bash

#SBATCH -A ctb-steinman
#SBATCH --job-name=coronary
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Requesting resources
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --time=23:59:00

# Export all environment vars
#SBATCH --export=ALL


module load NiaEnv/.2020a intel/2020u1 intelmpi/2020u1 intelpython3/2020u1 cmake/3.16.3 boost/1.69.0 eigen/3.3.7 hdf5/1.8.21 netcdf/4.6.3 gmp/6.2.0 mpfr/4.0.2 swig/4.0.1 petsc/3.10.5 trilinos/12.12.1 fenics/2019.1.0

source activate oasis

# Navigate to the script directory
cd $SLURM_SUBMIT_DIR

# Set writable directories for caching
export DIJITSO_CACHE_DIR=/scratch/s/steinman/ranbar/Torino/Coronary/scripts/.cache
export FFC_CACHE_DIR=/scratch/s/steinman/ranbar/Torino/Coronary/scripts/.ffc_cache
    
# Ensure the directories exist
mkdir -p $DIJITSO_CACHE_DIR
mkdir -p $FFC_CACHE_DIR

mpirun -n 80 oasis NSfracStep problem=coronary
#python ~/Oasis/oasis/run_oasis.py NSfracStep problem=coronary