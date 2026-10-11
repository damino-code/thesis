#!/bin/bash
#SBATCH --job-name=baseline_fou18
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h200-1,gpu-nvidia-h200-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/home/amine/thesis/baseline2000/logs/baseline_fou18_%j.out
#SBATCH --error=/storage/home/amine/thesis/baseline2000/logs/baseline_fou18_%j.err

echo "=========================================================="
echo "baseline2000 — Hate-Speech Classification on test_2000_fou18.csv"
echo "NOTE: run testRidge pipeline for this dataset first (attribute_aware_with_values depends on it)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

export DATASET=test_2000_fou18

cd /storage/home/amine/thesis/baseline2000

nvidia-smi

source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh
mamba activate llm-research

set -a; source /storage/home/amine/thesis/.env; set +a
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE_DIR=/storage/home/amine/.cache/torchinductor_autotune
mkdir -p /storage/home/amine/.cache/torchinductor_autotune

mkdir -p /storage/nobackup/amine/models
mkdir -p /storage/home/amine/thesis/baseline2000/logs

echo ""
echo "1. RUNNING ALL STRATEGIES (model loaded once)..."
echo -e "all\nall\n" | python src/run_classification.py

echo ""
echo "2. EVALUATING (all strategies side-by-side)..."
python src/evaluation.py

echo ""
echo "=========================================================="
echo "Done at $(date)"
