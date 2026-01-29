import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
THESIS_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

# Dataset
DATASET_FOLDER = os.path.join(THESIS_DIR, "Dataset")
DATASET_FILENAME = "selected_comments.csv"
DATASET_PATH = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

# Models (Qwen)
MODEL_FOLDER = "/scratch/amine/models"
REPO_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"
# Use the main Qwen GGUF file (update if you use a different quantization)
FILENAME = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
MODEL_PATH = os.path.join(MODEL_FOLDER, FILENAME)

# Results
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
VISUALIZATIONS_FOLDER = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)

# Analysis Config
ATTRIBUTES = [
    'sentiment', 'respect', 'insult', 'humiliate', 'status',
    'dehumanize', 'violence', 'genocide', 'attack_defend', 'hatespeech'
]

# Ensure directories exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)
