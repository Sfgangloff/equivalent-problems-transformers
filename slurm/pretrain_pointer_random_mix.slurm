#!/bin/bash
#SBATCH --job-name=sudoku_pretrain_pointer_randmix
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

# Cheap-option follow-up to pretrain_pointer_multi_alphabet.slurm: instead
# of K=6 FIXED alphabets (each puzzle repeated once per letter, so the
# model can partially memorize those 6 specific combinations), every
# training example gets an INDEPENDENTLY RANDOM digit-to-letter mapping
# drawn fresh from all 25 non-E letters (see RandomMixPointerDataset) --
# no fixed alphabet combination is ever repeated to memorize. Tests
# whether the K=6 result's partial zero-shot transfer gap (14.5% cell
# accuracy vs ~11% chance, 0% valid_rate on held-out E -- see
# results/eval_pointer_results.jsonl) is a training-diversity problem
# (should close under this) or something more structural.
#
# No n_puzzles repetition needed (unlike the K=6 job): each of the
# ~178,560 fully-represented train puzzles (99.2% of 180,000) gets a
# fresh random mix every epoch just by being visited, so this uses FEWER
# examples/epoch than the K=6 job (178,560 vs 360,000) despite covering
# far more combinations -- expect this to run faster than the ~69min
# K=6 pretrain, hence the 1.5h (not 2h) budget.
#
# Produces checkpoints/base_pointerRandomMix.pt, consumed by
# eval_pointer_zero_shot.slurm / eval_pointer_mixed.slurm (point
# --checkpoint at this file instead of base_pointerMultiABCDFG.pt).

source slurm/leonardo_env.sh
python -m src.equiv.train_pointer --config configs/base.yaml \
    --random-mix-letters A,B,C,D,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z \
    --epochs 20
