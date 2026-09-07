#!/bin/bash
#SBATCH --job-name=sudoku_eval_cross
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
# No --qos set: see train_alphabet_a.slurm. This job is quick (10 short
# eval passes) so dbg's 30min cap would probably be fine too, but there's
# no reason to risk the boundary.
#SBATCH --account=EUHPC_B38_121
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs after train_alphabet_a.slurm, train_alphabet_b.slurm, and
# train_union.slurm have all produced their checkpoints. Produces the
# headline results table: self-eval, naive cross-eval, and transplanted
# cross-eval for the A and B models, plus the union and random-init
# reference rows -- see README.md for what each number isolates.

source slurm/leonardo_env.sh

OUT=results/eval_results.jsonl
CFG=configs/base.yaml

run() { python -m src.equiv.evaluate --config "$CFG" --checkpoint "$1" --alphabet "$2" $3 --out "$OUT"; }

run checkpoints/base_A.pt A ""
run checkpoints/base_A.pt B ""
run checkpoints/base_A.pt B "--transplant"

run checkpoints/base_B.pt B ""
run checkpoints/base_B.pt A ""
run checkpoints/base_B.pt A "--transplant"

run checkpoints/base_union.pt A ""
run checkpoints/base_union.pt B ""

run checkpoints/base_A_random.pt A ""
run checkpoints/base_A_random.pt B ""

echo "wrote $OUT"
