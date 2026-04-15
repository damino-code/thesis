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
echo "Starting Llama 3.2 8B Analysis - DYNAMIC PROMPTS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

# Create logs directory if it doesn't exist
mkdir -p logs

cd /storage/home/amine/thesis/Llama-3.2-8B-Instruct-Q4_K_M/separate_hatespeech_multiattribute

# Check GPU availability
nvidia-smi

# Use SYSTEM CUDA 13.1 (from /usr/local/cuda) instead of conda CUDA 12.6 for driver compatibility
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda  
export CUDA_VERSION=13.1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/storage/home/amine/miniforge3/lib:$LD_LIBRARY_PATH
export FORCE_CUDA=1

echo "CUDA Environment:"
which nvcc
nvcc --version
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Source the Mamba/Conda initialization script so the 'mamba' command works in batch mode
source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh

# Activate your environment
mamba activate llm-research

# CRITICAL: Set LD_LIBRARY_PATH for CUDA runtime libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""
echo "1. RUNNING ALL ATTRIBUTES (Feature/Dynamic Mode)..."
# We pipe '2' (Dynamic mode) and 'all' (Sample size) to the run_all_attributes script
echo -e "2\nall\n" | python src/run_all_attributes.py

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
