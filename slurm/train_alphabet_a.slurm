#!/bin/bash
#SBATCH --job-name=sudoku_train_a
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg   # TODO: this QOS caps out at 30min/8 nodes -- fine for a first
                               #       smoke run, switch to the default QOS or boost_qos_lprod
                               #       (<=4 days) once that smoke run has passed.
#SBATCH --account=EUHPC_B38_121
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source slurm/leonardo_env.sh
srun python -m src.equiv.train --config configs/base.yaml --alphabet A
