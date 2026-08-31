#!/bin/bash
# Sourced by every slurm/*.slurm script to set up the Python/CUDA environment.
#
# Two supported setup paths on Leonardo (per CINECA docs, researched Aug
# 2026) -- pick ONE, comment out the other. I could NOT verify exact current
# module *version strings* from outside Leonardo (they drift over time).
#
#   Before submitting any real job, run on a Leonardo login node:
#       module av cineca-ai
#       module av python
#       module av cuda
#   and paste the output back so the placeholders below can be pinned to
#   real versions -- do not sbatch with the placeholders still in here.

set -euo pipefail

# --- Path 1 (recommended, less setup): pre-built PyTorch/CUDA stack -------
# module load profile/deeplrn
# module load cineca-ai/<TODO: pin from `module av cineca-ai`>
# python -m venv --system-site-packages "$HOME/envs/equiv" 2>/dev/null || true
# source "$HOME/envs/equiv/bin/activate"
# pip install pyyaml pytest  # cineca-ai already provides torch/numpy

# --- Path 2: manual modules + plain venv -----------------------------------
module load python/<TODO: pin from `module av python`>
module load cuda/<TODO: pin from `module av cuda`>
python -m venv "$HOME/envs/equiv" 2>/dev/null || true
source "$HOME/envs/equiv/bin/activate"
pip install -r requirements.txt

cd "$SLURM_SUBMIT_DIR"
