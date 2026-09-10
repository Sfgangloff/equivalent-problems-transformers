#!/bin/bash
#SBATCH --job-name=sudoku_eval_pointer_mixed
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

# Runs after pretrain_pointer_multi_alphabet.slurm (needs
# checkpoints/base_pointerMultiABCDFG.pt). Pure evaluation, no training.
#
# Pointer-model analogue of eval_mixed_alphabets.slurm: does the
# architectural fix (no fixed per-alphabet output vocabulary) also help
# with compositional generalization over KNOWN symbols recombined in
# novel ways -- distinct from, and an easier question than,
# eval_pointer_zero_shot.slurm's genuinely-novel-alphabet test. Same 2
# controls + 10 random mixes, same 1000-puzzle subset size, for direct
# comparison against the classifier model's results
# (results/eval_mixed_results.jsonl, mean cell_accuracy 45.4%).

source slurm/leonardo_env.sh
python -m src.equiv.evaluate_pointer_mixed --config configs/base.yaml \
    --checkpoint checkpoints/base_pointerMultiABCDFG.pt --letters A,B,C,D,F,G \
    --n-trials 10 --n-puzzles 1000 --split test --seed 0 \
    --out results/eval_pointer_mixed_results.jsonl

echo "wrote results/eval_pointer_mixed_results.jsonl"
