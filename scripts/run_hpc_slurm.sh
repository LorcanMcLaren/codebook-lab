#!/bin/bash -l
#SBATCH --job-name=codebook-lab
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

# Replace these with the modules and environment activation your cluster uses.
# module load anaconda3
# conda activate codebook-lab
# module load ollama

# Set country_iso_code in param_grid.yaml to the country where this cluster is
# physically located. That value is used for CodeCarbon emissions estimates.

bash "$REPO_ROOT/scripts/run_local.sh"
