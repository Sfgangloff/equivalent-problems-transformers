#!/bin/bash
#SBATCH --job-name=sudoku_eval_finetune
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_B38_121
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs after finetune_sweep.slurm. Evaluates every finetune/scratch
# checkpoint on alphabet E's FULL, uncapped test split (10k puzzles) -- the
# rigorous number, unlike the capped per-epoch monitoring accuracy logged
# during the sweep. Also records the free zero-shot floor: the
# multi-alphabet checkpoint evaluated directly on E with no fine-tuning at
# all (expect ~chance, per Experiment 0's random-init floor).

source slurm/leonardo_env.sh

CFG=configs/base.yaml
OUT=results/eval_finetune_results.jsonl

run() { python -m src.equiv.evaluate --config "$CFG" --checkpoint "$1" --alphabet E --out "$OUT"; }

# zero-shot floor: pretrained on 6 alphabets, E never seen, no fine-tuning
run checkpoints/base_multiABCDFG.pt

for n in 100 1000; do
  for seed in 0 1 2; do
    run "checkpoints/base_finetuneE_n${n}_seed${seed}.pt"
    run "checkpoints/base_scratchE_n${n}_seed${seed}.pt"
  done
done

echo "wrote $OUT"
