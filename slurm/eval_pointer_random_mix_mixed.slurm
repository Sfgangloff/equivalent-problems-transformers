#!/bin/bash
#SBATCH --job-name=sudoku_eval_pointer_randmix_mixed
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

# Runs after pretrain_pointer_random_mix.slurm. Mirrors
# eval_pointer_mixed.slurm but for checkpoints/base_pointerRandomMix.pt --
# same 2 controls + 10 random mixes over the SAME A,B,C,D,F,G letters used
# in the K=6 fixed-alphabet mixed-eval, for direct comparison against both
# earlier results (classifier: mean 45.4%; K=6 fixed-alphabet pointer:
# results/eval_pointer_mixed_results.jsonl).

source slurm/leonardo_env.sh
python -m src.equiv.evaluate_pointer_mixed --config configs/base.yaml \
    --checkpoint checkpoints/base_pointerRandomMix.pt --letters A,B,C,D,F,G \
    --n-trials 10 --n-puzzles 1000 --split test --seed 0 \
    --out results/eval_pointer_random_mix_mixed_results.jsonl

echo "wrote results/eval_pointer_random_mix_mixed_results.jsonl"
