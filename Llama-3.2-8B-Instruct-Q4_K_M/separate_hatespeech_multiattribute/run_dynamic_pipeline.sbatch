#!/bin/bash
#SBATCH --job-name=async_feature_run
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dynamic_%j.out
#SBATCH --error=logs/dynamic_%j.err

# Print basic job information
echo "=========================================================="
echo "Starting Llama 3.2 8B Analysis - DYNAMIC PROMPTS (ASYNC BATCHING)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

# Create logs directory if it doesn't exist
mkdir -p logs

cd /storage/home/amine/thesis/Llama-3.2-8B-Instruct-Q4_K_M/separate_hatespeech_multiattribute

# Check GPU availability
nvidia-smi

# Source the Mamba/Conda initialization script so the 'mamba' command works in batch mode
source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh

# Activate your environment
mamba activate llm-research

echo ""
echo "1. RUNNING ALL ATTRIBUTES (Feature/Dynamic Mode) USING ASYNC BATCHING..."
# We pipe '2' (Dynamic mode) and '3' (Sample size) to the new async batching script
echo -e "2\nall\n" | python src/async_batch_runner.py

echo ""
echo "2. MERGING RESULTS..."
# Pass "dynamic" argument to merge the persona_results folder instead of standard
python src/merge_results.py dynamic

echo ""
echo "3. EVALUATING..."
# Pipe '2' to select "Persona (Feature-based dynamic prompts)" in the evaluation menu
echo "2" | python src/evaluation.py

echo ""
echo "=========================================================="
echo "Analysis completely finished at $(date)"
