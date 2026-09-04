#!/bin/bash
# Sourced by every slurm/*.slurm script to set up the Python/CUDA environment.
#
# Confirmed working on Leonardo (2026-09-04, account EUHPC_B38_121, tested
# via `srun -p boost_usr_prod --qos boost_qos_dbg --gres=gpu:1 --pty bash`):
# `cineca-ai/4.1.1` (loaded via `profile/deeplrn`) provides torch 2.0.0a0
# with working CUDA (torch.cuda.is_available() -> True, 1 GPU visible per
# `--gres=gpu:1`). We layer just pyyaml/pytest on top via a
# --system-site-packages venv so we don't reinstall torch/numpy ourselves.

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/4.1.1

python3 -m venv --system-site-packages "$HOME/envs/equiv" 2>/dev/null || true
source "$HOME/envs/equiv/bin/activate"
pip install --quiet pyyaml pytest

cd "$SLURM_SUBMIT_DIR"
