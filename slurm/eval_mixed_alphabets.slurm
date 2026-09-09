#!/bin/bash
#SBATCH --job-name=sudoku_eval_mixed
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_B38_121
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs after pretrain_multi_alphabet.slurm (needs checkpoints/base_multiABCDFG.pt).
# Pure evaluation, no training -- can run independently of finetune_sweep.slurm.
#
# Tests compositional generalization: 2 controls (all-one-known-alphabet,
# sanity check -- should match ordinary self-eval; maximally-mixed-cycle,
# every digit from a different known alphabet) + 10 random digit-to-alphabet
# mixes, each on the SAME 1000-puzzle test subset. See evaluate_mixed.py's
# docstring for the full rationale (idea from a colleague: does the model
# bind each symbol's meaning independently of its co-occurring symbols, or
# did it just learn to recognize/route through one of the 6 known coherent
# alphabets as a whole?).

source slurm/leonardo_env.sh
python -m src.equiv.evaluate_mixed --config configs/base.yaml \
    --checkpoint checkpoints/base_multiABCDFG.pt --letters A,B,C,D,F,G \
    --n-trials 10 --n-puzzles 1000 --split test --seed 0 \
    --out results/eval_mixed_results.jsonl

echo "wrote results/eval_mixed_results.jsonl"
