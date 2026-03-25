#!/bin/bash
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --exclude=cn07
#SBATCH --job-name=iitj-run-all
#SBATCH --output=output_run_all_%j.log

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 /csehome/m25csa031/iitj_pipeline_final/run_all.py "$@"
