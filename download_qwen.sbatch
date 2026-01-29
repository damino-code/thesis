#!/bin/bash
#SBATCH -J qwen_download
#SBATCH -p compute
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH -o slurm-%A.out
#SBATCH -e slurm-%A.err

# Optional: activate your project venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Ensure a writable model directory (can override on submission)
export MODEL_DIR=${MODEL_DIR:-"/storage/home/amine/models"}

python download_qwen.py