#!/bin/bash
#SBATCH --job-name=sudoku_train_a
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
# No --qos set: uses the account's default QOS (~24h cap), since the
# 30min-capped boost_qos_dbg smoke test (interactive, tiny.yaml) already
# passed on 2026-09-07 and this job's --time=02:00:00 exceeds dbg's cap.
#SBATCH --account=EUHPC_XXXX_XXX
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Single GPU only: train.py has no distributed/DDP support, so requesting
# more (as an earlier version of this script did, with --ntasks-per-node=4
# --gres=gpu:4) launched 4 uncoordinated copies of the same run racing to
# write the same checkpoint/log path -- confirmed on Leonardo 2026-09-07
# (job hung with no progress for 1.5h+ before being cancelled). This model
# size (d_model=384, 8 layers, ~200k puzzles) doesn't need multi-GPU anyway.
source slurm/leonardo_env.sh
srun python -m src.equiv.train --config configs/base.yaml --alphabet A
