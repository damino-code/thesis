#!/bin/bash
#SBATCH --job-name=llama70b_vanilla
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h200-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/home/amine/thesis/FullAnnotation_LLama/logs/llama70b_vanilla_%j.out
#SBATCH --error=/storage/home/amine/thesis/FullAnnotation_LLama/logs/llama70b_vanilla_%j.err

echo "=========================================================="
echo "Starting Llama 3.1 70B AWQ Analysis (vLLM) - VANILLA PROMPTS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

cd /storage/home/amine/thesis/FullAnnotation_LLama

nvidia-smi

# Source Mamba
source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh
mamba activate llm-research

# Load environment variables (.env) and fix libstdc++
set -a; source /storage/home/amine/thesis/.env; set +a
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Ensure model download dir exists on shared storage
mkdir -p /storage/nobackup/amine/models

echo ""
echo "1. RUNNING ALL ATTRIBUTES (Vanilla Mode)..."
echo -e "1\nall\n" | python src/run_all_attributes.py

echo ""
echo "2. MERGING RESULTS..."
python src/merge_results.py

echo ""
echo "3. EVALUATING..."
echo "1" | python src/evaluation.py

echo ""
echo "=========================================================="
echo "Analysis completely finished at $(date)"
