#!/bin/bash
#SBATCH --job-name=sudoku_eval_pointer_randmix
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

# Runs after pretrain_pointer_random_mix.slurm. Mirrors
# eval_pointer_zero_shot.slurm but for checkpoints/base_pointerRandomMix.pt:
# gate on a few known letters (in-distribution, but note these were only
# ever seen as INGREDIENTS in random mixes, never as a coherent single
# alphabet during training -- itself informative), then the real test:
# zero-shot on held-out E, never in the random-mix pool at all.

source slurm/leonardo_env.sh

CFG=configs/base.yaml
CKPT=checkpoints/base_pointerRandomMix.pt
OUT=results/eval_pointer_random_mix_results.jsonl

run() { python -m src.equiv.evaluate_pointer --config "$CFG" --checkpoint "$CKPT" --alphabet "$1" --out "$OUT"; }

# gate: a few known (mix-ingredient) letters
for letter in A B C D F G; do
  run "$letter"
done

# the actual test: zero-shot on the held-out alphabet
run E

echo "wrote $OUT"
