import os


def _load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
THESIS_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

# Dataset
DATASET_FOLDER = os.path.join(THESIS_DIR, "Dataset")
DATASET_FILENAME = "selected_comments.csv"
DATASET_PATH = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

# Model (OpenRouter)
_load_dotenv(os.path.join(THESIS_DIR, ".env"))

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-next-80b-a3b-instruct")
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPEN_ROUTER_API_KEY")
)
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "thesis-attribute-analysis")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
DEFAULT_CONFIDENCE = float(os.getenv("OPENROUTER_DEFAULT_CONFIDENCE", "1.0"))

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
