import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

DATASET_FOLDER = os.path.join(THESIS_DIR, "Dataset")
DATASET_FILENAME = "processed_dataset.csv"
DATASET_PATH = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"
MODEL_DOWNLOAD_DIR = "/storage/nobackup/amine/models"

# vLLM
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 256
TENSOR_PARALLEL_SIZE = 1
SEED = 42

RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
VISUALIZATIONS_FOLDER = os.path.join(BASE_DIR, "visualizations")

# Source of LLM-predicted attribute values used by the
# attribute_aware_with_values prompt. Joined on (comment_id, annotator_id).
FULL_ANNOTATION_DIR = os.path.join(THESIS_DIR, "FullAnnotation_Qwen")
ATTRIBUTE_VALUES_PATTERN = os.path.join(
    FULL_ANNOTATION_DIR, "results", "merged_standard_results_*.csv"
)

# Ten attribute prompts — used by attribute_aware_with_values.
# Native scale max → used to bin a value into {low, moderate, high} as thirds.
ATTRIBUTE_SCALES = {
    "sentiment":     4,   # 0..4
    "respect":       4,   # 0..4
    "insult":        3,   # 0..3
    "humiliate":     2,   # 0..2
    "status":        1,   # 0..1
    "dehumanize":    1,   # 0..1
    "violence":      1,   # 0..1
    "genocide":      1,   # 0..1
    "attack_defend": 3,   # 0..3
    "hatespeech":    2,   # 0..2 — LLM scale {0=Yes,1=No,2=Unclear}; excluded from
                          # attribute_aware_with_values since it is the question.
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

# Four prompt strategies. Each maps to a JSON file in src/prompts/<name>.json.
PROMPT_STRATEGIES = [
    "zero_shot",
    "definition",
    "attribute_aware_no_values",
    "attribute_aware_with_values",
]

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)
