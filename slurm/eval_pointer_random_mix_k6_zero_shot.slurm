#!/bin/bash
#SBATCH --job-name=sudoku_eval_pointer_randmix_k6
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_XXXX_XXX
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs after pretrain_pointer_random_mix_k6.slurm. Gate on the 6 known
# (mix-ingredient) letters, then zero-shot on held-out E -- for
# completeness/consistency with Experiments 2 and 3, though the primary
# comparison for this disentangling experiment is the mixed-alphabet eval.

source slurm/leonardo_env.sh

CFG=configs/base.yaml
CKPT=checkpoints/base_pointerRandomMix_K6.pt
OUT=results/eval_pointer_random_mix_k6_results.jsonl

run() { python -m src.equiv.evaluate_pointer --config "$CFG" --checkpoint "$CKPT" --alphabet "$1" --out "$OUT"; }

for letter in A B C D F G; do
  run "$letter"
done
run E

echo "wrote $OUT"
