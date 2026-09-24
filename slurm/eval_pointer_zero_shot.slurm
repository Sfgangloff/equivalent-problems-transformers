#!/bin/bash
#SBATCH --job-name=sudoku_eval_pointer
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

# Runs after pretrain_pointer_multi_alphabet.slurm. Gate first: in-
# distribution valid_rate on the 6 KNOWN alphabets should be near the
# classifier model's ceiling (~90%+) -- a low zero-shot E number is
# uninterpretable without confirming the architecture learned to solve
# Sudoku at all. Then the actual test: zero-shot on held-out E, no
# fine-tuning, full unfiltered test split (the whole point of the
# architecture is not needing any of that).

source slurm/leonardo_env.sh

CFG=configs/base.yaml
CKPT=checkpoints/base_pointerMultiABCDFG.pt
OUT=results/eval_pointer_results.jsonl

run() { python -m src.equiv.evaluate_pointer --config "$CFG" --checkpoint "$CKPT" --alphabet "$1" --out "$OUT"; }

# gate: in-distribution on all 6 known alphabets
for letter in A B C D F G; do
  run "$letter"
done

# the actual test: zero-shot on the held-out alphabet, never seen in training
run E

echo "wrote $OUT"
