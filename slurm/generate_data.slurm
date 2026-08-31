#!/bin/bash
#SBATCH --job-name=sudoku_gen
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=dcgp_usr_prod   # TODO: confirm. Puzzle generation is CPU-only and needs
                                     #       no GPU -- dcgp_usr_prod is Leonardo's CPU partition
                                     #       per the docs, but confirm your project has access
                                     #       to it (or just run this directly on a login node,
                                     #       it's short and CPU-light).
#SBATCH --account=<TODO_project_account>
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/leonardo_env.sh
python scripts/generate_sudoku.py --config configs/base.yaml
