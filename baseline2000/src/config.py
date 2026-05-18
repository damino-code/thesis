import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

# Test dataset
DATASET_PATH = os.path.join(BASE_DIR, "..", "testRdige", "test_2000.csv")
DATASET_PATH = os.path.normpath(DATASET_PATH)

# Model — Llama-3.1-70B (same as the annotation run in testRdige)
MODEL_ID               = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
MODEL_DOWNLOAD_DIR     = "/storage/nobackup/amine/models"
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN          = 1024
MAX_NUM_SEQS           = 256
TENSOR_PARALLEL_SIZE   = 1
SEED                   = 42

RESULTS_FOLDER       = os.path.join(BASE_DIR, "results")
VISUALIZATIONS_FOLDER = os.path.join(BASE_DIR, "visualizations")

# Attribute annotations from the testRdige pipeline (used by attribute_aware_with_values)
TESTRIDGE_RESULTS_DIR    = os.path.normpath(os.path.join(BASE_DIR, "..", "testRdige", "results"))
ATTRIBUTE_VALUES_PATTERN = os.path.join(TESTRIDGE_RESULTS_DIR, "merged_annotations_*.csv")

ATTRIBUTE_SCALES = {
    "sentiment":     4,
    "respect":       4,
    "insult":        3,
    "humiliate":     2,
    "status":        1,
    "dehumanize":    1,
    "violence":      1,
    "genocide":      1,
    "attack_defend": 3,
    "hatespeech":    2,
}
ATTRIBUTE_HUMAN_LABELS = {
    "sentiment":     "Sentiment",
    "respect":       "Respect",
    "insult":        "Insult",
    "humiliate":     "Humiliate",
    "status":        "Derogatory Status",
    "dehumanize":    "Dehumanize",
    "violence":      "Violence",
    "genocide":      "Genocide",
    "attack_defend": "Attack/Defend",
}

PROMPT_STRATEGIES = [
    "zero_shot",
    "few_shot",
    "definition",
    "attribute_aware_no_values",
    "attribute_aware_with_values",
]

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)
