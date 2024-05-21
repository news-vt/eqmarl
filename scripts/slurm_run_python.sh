#!/bin/bash
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 48
#SBATCH --time=48:00:00

# Load modules.
echo "Loading modules"
module reset >/dev/null 2>&1
module load Anaconda3 >/dev/null 2>&1

# Build anaconda environment name from SLURM allocation.
CONDA_ENV_NAME="eqmarl"

# Initialize the shell to use Anaconda.
eval "$(conda shell.bash hook)"

# Activate Anaconda environment.
conda activate ${CONDA_ENV_NAME}

# Propagate arguments to Python and run.
python $@