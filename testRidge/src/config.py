import os
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
THESIS_DIR = BASE_DIR.parent

DATASET = os.environ.get("DATASET", "test_2000")

# Test dataset
TEST_CSV = THESIS_DIR / "testDataset" / f"{DATASET}.csv"

# Annotation model (Llama-3.1-70B vanilla — best AUROC on training set: 0.8988)
MODEL_ID               = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
MODEL_DOWNLOAD_DIR     = "/storage/nobackup/amine/models"
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN          = 1024
MAX_NUM_SEQS           = 256
TENSOR_PARALLEL_SIZE   = 1
SEED                   = 42

ATTRIBUTES = [
    "sentiment", "respect", "insult", "humiliate", "status",
    "dehumanize", "violence", "genocide", "attack_defend", "hatespeech",
]

# Reuse prompt files from the Llama annotation pipeline
PROMPTS_DIR = THESIS_DIR / "FullAnnotation_LLama" / "src" / "prompts"

# Saved Ridge weights (vanilla mode, Llama-3.1-70B)
RIDGE_WEIGHTS_PATH = (
    THESIS_DIR / "HateSpeechScoreFull" / "Llama-3.1-70B"
    / "results" / "standard" / "weights.json"
)

# Output dirs
RESULTS_DIR       = BASE_DIR / "results" / DATASET
ATTR_RESULTS_DIR  = RESULTS_DIR / "per_attribute"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ATTR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
