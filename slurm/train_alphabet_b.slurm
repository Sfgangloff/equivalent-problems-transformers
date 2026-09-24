#!/bin/bash
#SBATCH --job-name=sudoku_train_b
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
# No --qos set: see train_alphabet_a.slurm
#SBATCH --account=EUHPC_XXXX_XXX
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Single GPU only: see train_alphabet_a.slurm
source slurm/leonardo_env.sh
srun python -m src.equiv.train --config configs/base.yaml --alphabet B
