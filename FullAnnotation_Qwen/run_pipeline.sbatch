#!/bin/bash
#SBATCH --job-name=qwen72b_vllm
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h200-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=/storage/home/amine/thesis/FullAnnotation_Qwen/logs/qwen72b_%j.out
#SBATCH --error=/storage/home/amine/thesis/FullAnnotation_Qwen/logs/qwen72b_%j.err

echo "=========================================================="
echo "Starting Qwen2.5 72B AWQ Analysis (vLLM) - DYNAMIC PROMPTS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

cd /storage/home/amine/thesis/FullAnnotation_Qwen

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
echo "1. RUNNING ALL ATTRIBUTES (Feature/Dynamic Mode)..."
echo -e "2\nall\n" | python src/run_all_attributes.py

echo ""
echo "2. MERGING RESULTS..."
python src/merge_results.py dynamic

echo ""
echo "3. EVALUATING..."
echo "2" | python src/evaluation.py

echo ""
echo "=========================================================="
echo "Analysis completely finished at $(date)"
