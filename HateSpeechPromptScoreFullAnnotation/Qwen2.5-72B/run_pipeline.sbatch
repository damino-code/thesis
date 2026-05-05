#!/bin/bash
#SBATCH --job-name=qwen72b_hsscore_persona
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h200-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/storage/home/amine/thesis/HateSpeechPromptScoreFullAnnotation/Qwen2.5-72B/logs/qwen72b_hsscore_persona_%j.out
#SBATCH --error=/storage/home/amine/thesis/HateSpeechPromptScoreFullAnnotation/Qwen2.5-72B/logs/qwen72b_hsscore_persona_%j.err

echo "=========================================================="
echo "Qwen2.5-72B AWQ — Hate-Speech-Score Prompt (PERSONA)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

cd /storage/home/amine/thesis/HateSpeechPromptScoreFullAnnotation/Qwen2.5-72B

nvidia-smi

source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh
mamba activate llm-research

set -a; source /storage/home/amine/thesis/.env; set +a
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p /storage/nobackup/amine/models

echo ""
echo "1. RUNNING HATE-SPEECH-SCORE PROMPT (Persona Mode)..."
echo -e "2\nall\n" | python src/run_score.py

echo ""
echo "2. EVALUATING..."
printf '2\n' | python src/evaluation.py

echo ""
echo "=========================================================="
echo "Analysis completely finished at $(date)"
