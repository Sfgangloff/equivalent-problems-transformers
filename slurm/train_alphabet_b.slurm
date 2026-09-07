#!/bin/bash
#SBATCH --job-name=sudoku_train_b
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
# No --qos set: see train_alphabet_a.slurm
#SBATCH --account=EUHPC_B38_121
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/leonardo_env.sh
srun python -m src.equiv.train --config configs/base.yaml --alphabet B
