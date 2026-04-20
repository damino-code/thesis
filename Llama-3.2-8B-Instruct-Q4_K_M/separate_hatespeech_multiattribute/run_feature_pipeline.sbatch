#!/bin/bash
#SBATCH --job-name=llama8b_feature
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-nvidia-h200-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/feature_%j.out
#SBATCH --error=logs/feature_%j.err

echo "=========================================================="
echo "Llama 3.2 8B - FEATURE/PERSONA PROMPTS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

mkdir -p logs

cd /storage/home/amine/thesis/Llama-3.2-8B-Instruct-Q4_K_M/separate_hatespeech_multiattribute

nvidia-smi

# Source Mamba
source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh
mamba activate llm-research

# Fix libstdc++ and CUDA
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

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
echo "Finished at $(date)"
