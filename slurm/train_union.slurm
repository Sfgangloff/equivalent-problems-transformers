#!/bin/bash
#SBATCH --job-name=sudoku_train_union
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
# No --qos set: see train_alphabet_a.slurm
#SBATCH --account=EUHPC_B38_121
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Single GPU only: see train_alphabet_a.slurm. --time is 3h (vs 2h for A/B)
# since the union dataset is 2x the size (both alphabets' views concatenated).
#
# Mixed-alphabet reference model (upper-bound baseline): sees both A and B
# during training, so cross-alphabet accuracy here is expected to be high --
# it's the ceiling the A-only/B-only models' transplanted cross-eval is
# compared against.
source slurm/leonardo_env.sh
srun python -m src.equiv.train --config configs/base.yaml --alphabet union

# Random-init floor baseline (untrained model, same architecture): cheap to
# add here since it reuses the same environment/allocation.
python -m src.equiv.train --config configs/base.yaml --alphabet A --epochs 0 --tag random
