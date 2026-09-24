#!/bin/bash
#SBATCH --job-name=sudoku_pretrain_pointer_randmix_k6
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_XXXX_XXX
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Disentangling experiment: pretrain_pointer_random_mix.slurm (Experiment 3)
# changed TWO things at once relative to pretrain_pointer_multi_alphabet.slurm
# (Experiment 2) -- pool size (6 -> 25 letters) AND whether training puzzles
# ever mix multiple letters within one puzzle (never -> always). This run
# holds pool size fixed at Experiment 2's level (the SAME 6 letters,
# A,B,C,D,F,G -- same vocab_size_for(6)=55 too) while giving it Experiment
# 3's always-mixed training format, to isolate which of the two variables
# is actually responsible for Experiment 3's dramatic drop in trial-to-
# trial variance on the mixed-alphabet compositional eval.
#
# --tag K6 disambiguates the checkpoint name from Experiment 3's
# checkpoints/base_pointerRandomMix.pt -> produces
# checkpoints/base_pointerRandomMix_K6.pt.

source slurm/leonardo_env.sh
python -m src.equiv.train_pointer --config configs/base.yaml \
    --random-mix-letters A,B,C,D,F,G \
    --epochs 20 --tag K6
