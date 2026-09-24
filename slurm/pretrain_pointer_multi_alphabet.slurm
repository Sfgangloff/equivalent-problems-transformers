#!/bin/bash
#SBATCH --job-name=sudoku_pretrain_pointer
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_XXXX_XXX
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Experiment 2: pointer/copy architecture (model/pointer_transformer.py).
# Same K=6/N=60,000-per-alphabet/20-epoch protocol as the classifier's
# multi-alphabet pretrain (pretrain_multi_alphabet.slurm), same held-out
# alphabet E, so the two architectures' numbers are directly comparable.
# Est. ~69min, same rate as the classifier pretrain (per-example cost is
# encoder-dominated either way).
#
# Produces checkpoints/base_pointerMultiABCDFG.pt, consumed by
# eval_pointer_zero_shot.slurm.

source slurm/leonardo_env.sh
python -m src.equiv.train_pointer --config configs/base.yaml \
    --alphabets A,B,C,D,F,G --n-puzzles 60000 --epochs 20
