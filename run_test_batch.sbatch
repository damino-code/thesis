#!/bin/bash
#SBATCH --job-name=test_batch_llm
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1      # Request 1 Nvidia GPU 
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --mem=32G                # Request 32GB of system RAM
#SBATCH --time=00:15:00          # Maximum time limit (15 minutes is plenty for this test)
#SBATCH --output=result_%j.out   # Standard output log file (%j = Job ID)
#SBATCH --error=result_%j.err    # Standard error log file (%j = Job ID)

# 1. Load CUDA module (Uncomment if your cluster requires explicit module loading)
# module load cuda

# 2. Source the Mamba/Conda initialization script so the 'mamba' command works in batch mode
source /storage/home/amine/miniforge3/etc/profile.d/conda.sh
source /storage/home/amine/miniforge3/etc/profile.d/mamba.sh

# 3. Activate your environment
mamba activate llm-research

# 4. Run the Python batching script
echo "Starting batch test on GPU node: $HOSTNAME"
python test_batch.py
echo "Job finished!"