#!/bin/bash
#SBATCH --job-name=llama70b_hsclassif
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h200-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/storage/home/amine/thesis/HateSpeechClassificationPromptFullAnnotation/Llama-3.3-70B/logs/llama70b_hsclassif_%j.out
#SBATCH --error=/storage/home/amine/thesis/HateSpeechClassificationPromptFullAnnotation/Llama-3.3-70B/logs/llama70b_hsclassif_%j.err

echo "=========================================================="
echo "Llama-3.3-70B AWQ — Hate-Speech Classification (4 prompts)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

cd /storage/home/amine/thesis/HateSpeechClassificationPromptFullAnnotation/Llama-3.3-70B

nvidia-smi

source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh
mamba activate llm-research

set -a; source /storage/home/amine/thesis/.env; set +a
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p /storage/nobackup/amine/models

echo ""
echo "1. RUNNING ALL 4 PROMPT STRATEGIES (model loaded once)..."
echo -e "all\nall\n" | python src/run_classification.py

echo ""
echo "2. EVALUATING (all strategies side-by-side)..."
python src/evaluation.py

echo ""
echo "=========================================================="
echo "Analysis completely finished at $(date)"
