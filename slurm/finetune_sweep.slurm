#!/bin/bash
#SBATCH --job-name=sudoku_finetune_sweep
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_B38_121
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs after pretrain_multi_alphabet.slurm has produced
# checkpoints/base_multiABCDFG.pt. Few-shot adaptation sweep: n in
# {100, 1000} x mode in {finetune, scratch} x 3 seeds = 12 runs, testing
# whether multi-alphabet pretraining lets the model adapt to held-out
# alphabet E from less data than training from scratch does. Each run is
# cheap (tiny train set, val capped to 500 puzzles for per-epoch
# monitoring -- see finetune.py) so all 12 run sequentially in one job;
# --mode finetune/--mode scratch with the same seed see the identical
# puzzle subset (fair comparison, see finetune.py's _n_puzzle_subset).
#
# The rigorous final numbers (full, uncapped test split) come from
# eval_finetune_results.slurm afterward, not the per-epoch logs here.

source slurm/leonardo_env.sh

CFG=configs/base.yaml
CKPT=checkpoints/base_multiABCDFG.pt
EPOCHS=150

for n in 100 1000; do
  for seed in 0 1 2; do
    python -m src.equiv.finetune --config "$CFG" --mode finetune \
        --checkpoint "$CKPT" --target-alphabet E \
        --n-puzzles "$n" --epochs "$EPOCHS" --seed "$seed"
    python -m src.equiv.finetune --config "$CFG" --mode scratch \
        --vocab-k 7 --target-alphabet E \
        --n-puzzles "$n" --epochs "$EPOCHS" --seed "$seed"
  done
done

echo "finetune sweep complete"
