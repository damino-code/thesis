#!/bin/bash
#SBATCH --job-name=llama70b_persona_remaining
#SBATCH --partition=compute
#SBATCH --exclude=gpu-nvidia-h200-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/home/amine/thesis/FullAnnotation_LLama/logs/llama70b_persona_remaining_%j.out
#SBATCH --error=/storage/home/amine/thesis/FullAnnotation_LLama/logs/llama70b_persona_remaining_%j.err

echo "=========================================================="
echo "Starting Llama 3.1 70B AWQ - PERSONA - hatespeech + attack_defend"
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

mkdir -p /storage/nobackup/amine/models

echo ""
echo "1. RUNNING hatespeech (Persona Mode)..."
python -c "
import sys
sys.path.insert(0, 'src')
from single_attribute_analyser.shared_runner import run_attribute_analysis
run_attribute_analysis('hatespeech', sample_size='all', use_dynamic=True)
"

echo ""
echo "2. RUNNING attack_defend (Persona Mode)..."
python -c "
import sys
sys.path.insert(0, 'src')
from single_attribute_analyser.shared_runner import run_attribute_analysis
run_attribute_analysis('attack_defend', sample_size='all', use_dynamic=True)
"

echo ""
echo "3. MERGING RESULTS..."
python src/merge_results.py dynamic

echo ""
echo "4. EVALUATING..."
printf '2\n\n' | python src/evaluation.py

echo ""
echo "=========================================================="
echo "Finished at $(date)"
