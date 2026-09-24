#!/bin/bash
#SBATCH --job-name=sudoku_eval_pointer_randmix_k6_mixed
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_XXXX_XXX
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs after pretrain_pointer_random_mix_k6.slurm. THE key comparison for
# the disentangling experiment: same 2 controls + 10 random mixes over the
# same A,B,C,D,F,G letters and same seed as every prior mixed-alphabet
# eval (classifier: mean 45.4%; K=6-fixed pointer: mean 52.8%;
# K=25-random-mix pointer: mean 98.2%, range collapsed to ~0.3pp). This
# run's range/variance tells us whether pool size (this run keeps it at
# 6, same as the fixed-alphabet condition) or mixing-exposure (this run
# has it, same as the K=25 condition) is the variable actually responsible
# for that variance collapse.

source slurm/leonardo_env.sh
python -m src.equiv.evaluate_pointer_mixed --config configs/base.yaml \
    --checkpoint checkpoints/base_pointerRandomMix_K6.pt --letters A,B,C,D,F,G \
    --n-trials 10 --n-puzzles 1000 --split test --seed 0 \
    --out results/eval_pointer_random_mix_k6_mixed_results.jsonl

echo "wrote results/eval_pointer_random_mix_k6_mixed_results.jsonl"
