import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

# Dataset
DATASET_FOLDER = os.path.join(THESIS_DIR, "Dataset")
DATASET_FILENAME = "processed_dataset.csv"
DATASET_PATH = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

# Model — vLLM uses HuggingFace model IDs (downloaded automatically)
MODEL_ID = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
MODEL_DOWNLOAD_DIR = "/storage/nobackup/amine/models"

# vLLM settings
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 256
TENSOR_PARALLEL_SIZE = 1
SEED = 42

# Results / outputs
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
VISUALIZATIONS_FOLDER = os.path.join(BASE_DIR, "visualizations")

# Hate-speech-score prompting config
ATTRIBUTE = "hate_speech_score"

# 0-9 bucket scale → continuous IRT-aligned score via shift = -4.5
# digit 0 → score -4.5  ... digit 9 → score +4.5
SCALE_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
SCALE_SHIFT = 4.5  # subtracted from digit to align with IRT (~[-5, +5])

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)
