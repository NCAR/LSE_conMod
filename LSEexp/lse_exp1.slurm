#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=lse_cfe
#SBATCH --cpus-per-task=36
#SBATCH --time=48:00:00
#SBATCH --output=/home/m10870071/scratch/logs/cfe/%j.out  # Standard output
#SBATCH --error=/home/m10870071/scratch/logs/cfe/%j.err   # Standard error
#SBATCH --account=2306081036    # Account number

ncpu=36

for iter in {0..5}
do
    echo "Starting iteration $iter" >&2
    python main_LSEworkflow_CFE.py config_CFEexp1.toml $iter $ncpu
done
