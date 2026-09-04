#!/bin/bash
#SBATCH --job-name=sudoku_gen
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_B38_121
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Puzzle generation is pure-Python/CPU (no GPU used) and single-threaded --
# ~14ms/puzzle measured locally, so configs/base.yaml's 200k puzzles is
# roughly 45min, hence the 1h budget. We didn't confirm access to Leonardo's
# CPU-only partition (dcgp_usr_prod) for this account, so this requests a
# single GPU on boost_usr_prod just to guarantee scheduling works -- the GPU
# itself sits idle. Wasteful of a fraction of a GPU-hour but low-risk; swap
# to a confirmed CPU partition later if you want to reclaim that budget.
source slurm/leonardo_env.sh
python scripts/generate_sudoku.py --config configs/base.yaml
