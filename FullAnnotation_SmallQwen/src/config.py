import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_DIR = os.path.dirname(BASE_DIR)

# Dataset
DATASET_FOLDER = os.path.join(THESIS_DIR, "Dataset")
DATASET_FILENAME = "processed_dataset.csv"
DATASET_PATH = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

# Model — vLLM uses HuggingFace model IDs (downloaded automatically)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MODEL_QUANTIZATION = "awq"
MODEL_DOWNLOAD_DIR = "/storage/nobackup/amine/models"

# vLLM settings
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 1024            # Prompts are ~300 tokens max; 1024 gives headroom
MAX_NUM_SEQS = 256              # Max concurrent sequences in vLLM scheduler
TENSOR_PARALLEL_SIZE = 1
SEED = 42                       # Fixed seed for reproducibility

# Results
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
VISUALIZATIONS_FOLDER = os.path.join(BASE_DIR, "visualizations")

# Analysis Config
ATTRIBUTES = [
    'sentiment', 'respect', 'insult', 'humiliate', 'status',
    'dehumanize', 'violence', 'genocide', 'attack_defend', 'hatespeech'
]

# Ensure directories exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)
