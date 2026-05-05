#!/bin/bash
#SBATCH --job-name=llama70b_hsscore_vanilla
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h200-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/home/amine/thesis/HateSpeechPromptScoreFullAnnotation/Llama-3.3-70B/logs/llama70b_hsscore_vanilla_%j.out
#SBATCH --error=/storage/home/amine/thesis/HateSpeechPromptScoreFullAnnotation/Llama-3.3-70B/logs/llama70b_hsscore_vanilla_%j.err

echo "=========================================================="
echo "Llama-3.3-70B AWQ — Hate-Speech-Score Prompt (VANILLA)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

cd /storage/home/amine/thesis/HateSpeechPromptScoreFullAnnotation/Llama-3.3-70B

nvidia-smi

source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh
mamba activate llm-research

set -a; source /storage/home/amine/thesis/.env; set +a
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p /storage/nobackup/amine/models

echo ""
echo "1. RUNNING HATE-SPEECH-SCORE PROMPT (Vanilla Mode)..."
echo -e "1\nall\n" | python src/run_score.py

echo ""
echo "2. EVALUATING..."
printf '1\n' | python src/evaluation.py

echo ""
echo "=========================================================="
echo "Analysis completely finished at $(date)"
