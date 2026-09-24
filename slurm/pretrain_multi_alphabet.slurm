#!/bin/bash
#SBATCH --job-name=sudoku_pretrain_multi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_XXXX_XXX
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# K=6 multi-alphabet pretraining (A,B,C,D,F,G -- E is deliberately skipped
# and reserved held-out for the few-shot/zero-shot follow-up experiments;
# the model is still sized to cover A..G so E's rows exist, untrained --
# see train.py's non-contiguous-letters vocab sizing).
#
# N=60,000 puzzles/alphabet (1/3 subset of the 180k train split, same
# indices reused across all 6 letters), 20 epochs. Est. ~69min based on the
# measured ~0.57ms/puzzle-epoch rate from the original train_A/train_union
# runs (results/2026-09-07_two_alphabet_sudoku.md), hence the 2h budget.
#
# Produces checkpoints/base_multiABCDFG.pt, consumed by
# finetune_sweep.slurm and eval_finetune_results.slurm.

source slurm/leonardo_env.sh
python -m src.equiv.train --config configs/base.yaml \
    --alphabets A,B,C,D,F,G --n-puzzles 60000 --epochs 20
