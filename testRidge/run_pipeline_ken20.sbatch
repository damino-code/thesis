#!/bin/bash
#SBATCH --job-name=ridge_ken20
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h200-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/home/amine/thesis/testRidge/logs/ridge_ken20_%j.out
#SBATCH --error=/storage/home/amine/thesis/testRidge/logs/ridge_ken20_%j.err

echo "=========================================================="
echo "Ridge Test — test_2000_ken20.csv  (Llama-3.1-70B vanilla)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

export DATASET=test_2000_ken20

cd /storage/home/amine/thesis/testRidge

nvidia-smi

source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh
mamba activate llm-research

set -a; source /storage/home/amine/thesis/.env; set +a
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p /storage/nobackup/amine/models
mkdir -p /storage/home/amine/thesis/testRidge/logs

echo ""
echo "1. ANNOTATING test_2000_ken20.csv (10 attributes, vanilla)..."
python src/run_annotation.py

echo ""
echo "2. MERGING PER-ATTRIBUTE RESULTS..."
python src/merge_results.py

echo ""
echo "3. APPLYING RIDGE WEIGHTS + EVALUATING..."
python src/evaluate_ridge.py

echo ""
echo "=========================================================="
echo "Done at $(date)"
