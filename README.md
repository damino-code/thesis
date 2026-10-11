# Hate Speech Analysis using Large Language Models

A comprehensive research project analyzing hate speech in text using state-of-the-art Large Language Models (LLMs) with annotation and evaluation pipelines.

## 📋 Project Overview

This project leverages multiple LLMs to analyze and annotate comments for hate speech characteristics. It evaluates different models (Llama, Qwen) at different scales (7B, 70B, 72B parameters) and uses Ridge regression for predictive modeling.

### Key Objectives
- Analyze comments for hate speech attributes (sentiment, respect, insults, humiliation, dehumanization, etc.)
- Compare performance across different LLM architectures and model sizes
- Build predictive models using Ridge regression
- Generate comprehensive visualizations and evaluation reports

---

## 🎯 What It Does

### Analysis Pipeline
1. **Attribute Annotation**: Uses LLMs to annotate comments across multiple dimensions:
   - **Sentiment**: Emotional tone (strongly negative → strongly positive, 5 levels)
   - **Respect**: Whether comment shows respect to targeted group (5 levels)
   - **Insult**: Presence of insulting language (4 levels)
   - **Humiliation**: Whether comment humiliates the group (3 levels)
   - **Status**: Framing groups as inferior/superior (2 levels)
   - **Dehumanization**: Portrayal of group as less than human (2 levels)

2. **Model Evaluation**: 
   - Tests annotations with and without persona prompts
   - Compares results across model architectures and sizes
   - Evaluates annotation quality and consistency

3. **Ridge Regression Modeling**: 
   - Trains predictive models on annotated data
   - Tests on various dataset sizes (2000, full dataset)
   - Generates performance metrics and visualizations

4. **Visualization & Reporting**:
   - Global comparisons across models
   - Per-model detailed results
   - Correlation analysis and metric reports

---

## 🤖 Models

### Large Models (70B+ parameters)
- **Llama**: Meta-Llama-3.3-70B-Instruct
- **Qwen**: Qwen2.5-72B-Instruct

### Small Models (7B-8B parameters)
- **Llama**: Meta-Llama-3.2-8B-Instruct (Q4_K_M quantized)
- **Qwen**: Qwen2.5-7B-Instruct (Q4_K_M quantized)

### Special Models
- **Qwen 3**: Qwen3-next-80b-a3b-instruct (experimental)

---

## 💾 Dataset

**Location**: `./Dataset/`
- **Source**: `selected_comments.csv` - Preprocessed hate speech comments
- **Format**: CSV with text columns and annotations
- **Size**: Full dataset ~2000+ comments (split versions available)

### Dataset Structure
```
{
  "text": "comment content",
  "sentiment": 1-5,
  "respect": 1-5,
  "insult": 1-4,
  "humiliate": 1-3,
  "status": 1-2,
  "dehumanize": 1-2
}
```

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.10+
- CUDA 12+ capable GPU (for GPU acceleration)
- 64GB+ RAM (for large models)
- ~100GB disk space (for models in GGUF format)

### 1. Clone Repository and Navigate
```bash
git clone https://github.com/damino-code/thesis.git
cd thesis
```

### 2. Create Environment
```bash
# Using Mamba (recommended)
mamba env create -f environment.yml
mamba activate llm-research

# Or using Conda
conda env create -f environment.yml
conda activate llm-research
```

### 3. Install Dependencies
```bash
pip install llama-cpp-python
pip install ollama
pip install openai
```

### 4. Check Setup
```bash
python check_setup.py
```

This will verify:
- ✅ Required Python libraries
- ✅ GPU availability (nvidia-smi)
- ✅ Dataset files
- ✅ Model paths

### 5. Configure Environment Variables
Create or update `.env` file:
```bash
export HF_TOKEN="your_huggingface_token"
export MODEL_PATH="/path/to/models"
export SCRATCH_PATH="/path/to/scratch"
```

---

## 📁 Project Structure

```
thesis/
├── README.md                              # This file
├── environment.yml                         # Python dependencies
├── check_setup.py                          # Verify setup
├── attributes_definition.txt               # Detailed attribute definitions
│
├── Dataset/
│   ├── selected_comments.csv               # Main dataset
│   ├── processed_dataset.csv               # Processed version
│   └── remap_hatespeech.py                 # Data preprocessing script
│
├── FullAnnotation_LLama/                  # Llama 70B results & pipeline
│   ├── src/
│   ├── visualizations/
│   ├── run_pipeline.sbatch                 # With persona prompts
│   └── run_vanilla_pipeline.sbatch         # Standard prompts
│
├── FullAnnotation_SmallLlama/             # Llama 8B results & pipeline
│   ├── src/
│   ├── visualizations/
│   ├── run_pipeline.sbatch
│   └── run_vanilla_pipeline.sbatch
│
├── FullAnnotation_Qwen/                   # Qwen 72B results & pipeline
│   ├── src/
│   ├── visualizations/
│   ├── run_pipeline.sbatch
│   └── run_vanilla_pipeline.sbatch
│
├── FullAnnotation_SmallQwen/              # Qwen 7B results & pipeline
│   ├── src/
│   ├── visualizations/
│   ├── run_pipeline.sbatch
│   └── run_vanilla_pipeline.sbatch
│
├── HateSpeechClassificationPromptFullAnnotation/
│   └── [Full annotation classification results by model]
│
├── testRdige/                             # Ridge regression tests
│   └── src/
│
├── global_visualisation/                  # Comparative visualizations
│   ├── Llama-3.2-8B/
│   ├── Llama-3.3-70B/
│   ├── Qwen2.5-7B/
│   ├── Qwen2.5-72B/
│   └── qwen3-next-80b/
│
├── examples/                               # Result examples
│   ├── ridge_results_*.txt                # Ridge regression results
│   ├── llama_*_metrics.txt                # Performance metrics
│   └── qwen_*_metrics.txt
│
└── [Model directories]
    ├── Llama-3.2-8B-Instruct-Q4_K_M/
    ├── Llama-3.3-70B-Instruct/
    ├── Qwen2.5-7B-Instruct-Q4_K_M/
    ├── Qwen2.5-72B-Instruct/
    └── qwen3-next-80b-a3b-instruct/
```

---

## 🚀 Running the Analysis

### Option 1: Local Execution (CPU/GPU)
```bash
# Verify setup first
python check_setup.py

# Run specific model analysis (e.g., Llama 70B)
cd FullAnnotation_LLama
python src/run_all_attributes.py
python src/merge_results.py
python src/evaluation.py
```

### Option 2: HPC Cluster (SLURM)
```bash
# Submit Llama 70B job with persona prompts
sbatch FullAnnotation_LLama/run_pipeline.sbatch

# Or without persona prompts (vanilla)
sbatch FullAnnotation_LLama/run_vanilla_pipeline.sbatch

# Submit Qwen 72B job
sbatch FullAnnotation_Qwen/run_pipeline.sbatch

# Check job status
squeue -u $USER

# View logs
tail -f FullAnnotation_LLama/logs/llama70b_*.out
```

### Option 3: Ridge Regression Analysis
```bash
cd testRdige
# Tests Ridge regression models on various configurations
python src/ridge_analysis.py
```

---

## 📊 Pipeline Stages

### Stage 1: Attribute Annotation
- Load LLM model (vLLM or llama-cpp-python)
- Iterate through dataset comments
- Generate LLM prompts with/without persona
- Extract annotation values for each attribute
- Save raw results

### Stage 2: Result Merging
- Combine results from all annotation runs
- Standardize formats and handle missing values
- Create consolidated annotation file

### Stage 3: Evaluation
- Compare annotations against baseline annotations
- Calculate agreement metrics (Accuracy, F1, etc.)
- Generate performance reports
- Identify model strengths/weaknesses per attribute

### Stage 4: Visualization
- Create comparison plots across models
- Generate heatmaps and distribution charts
- Produce statistical summaries

---

## 📈 Expected Outputs

### Per-Model Results
Each model folder generates:
- `results/merged_results_*.csv` - All annotations
- `logs/*.out` - SLURM stdout
- `logs/*.err` - SLURM stderr
- `visualizations/` - Charts and plots

### Global Comparisons
`global_visualisation/` contains:
- Model performance comparisons
- Attribute-level analysis by model
- Correlation heatmaps
- Ranking comparisons

### Ridge Regression Results
`examples/ridge_results_*.txt` contains:
- Training/test accuracy
- Cross-validation scores
- Feature importance
- Hyperparameter information

---

## ⚙️ Configuration

### Model Selection
Edit individual `run_pipeline.sbatch` files to change:
- Model path and size
- GPU allocation
- Memory limits
- Timeout values

### Dataset Configuration
In pipeline scripts:
```python
TEXT_COLUMN = "text"           # Column name with comments
SAMPLE_SIZE = None              # Use all data (or set integer)
SEED = 42                       # Reproducibility
```

### Prompt Engineering
Modify prompt templates in `src/prompts.py`:
- System prompts
- Few-shot examples
- Persona definitions
- Output format specifications

---

## 🔧 Troubleshooting

### GPU Out of Memory
- Reduce batch size in pipeline config
- Use quantized models (Q4_K_M)
- Use smaller model (7B instead of 70B)

### Slow Inference
- Ensure CUDA is properly set up: `nvidia-smi`
- Check GPU is being used: monitor `nvidia-smi` during run
- Use vLLM for better throughput

### Missing Models
Run setup script which will download missing models:
```bash
python check_setup.py --download-models
```

### Dataset Issues
Verify dataset format and paths:
```bash
python -c "import pandas as pd; df = pd.read_csv('Dataset/selected_comments.csv'); print(df.head())"
```

---

## 📚 References

### Papers & Models
- [Meta Llama 3 Models](https://huggingface.co/meta-llama)
- [Qwen Models](https://huggingface.co/Qwen)
- [Hate Speech Datasets](https://huggingface.co/datasets)

### Tools Used
- **llama-cpp-python**: Local LLM inference
- **vLLM**: Optimized LLM serving
- **scikit-learn**: Ridge regression & metrics
- **pandas/numpy**: Data processing
- **matplotlib/seaborn**: Visualization

---

## 📝 License

This research project is part of a thesis. Please include appropriate attribution if used.

---

## 👤 Author

**Amine** - LLM Research & Analysis

---

## 📞 Support

For issues or questions:
1. Check logs in model-specific `logs/` directories
2. Run `check_setup.py` to verify environment
3. Review error messages in SLURM `.err` files
4. Check `attributes_definition.txt` for annotation guidelines

---

## 🔄 Version History

- **v1.0** (Current): Full annotation pipeline with Llama & Qwen models
- **v0.9**: Initial Ridge regression implementation
- **v0.5**: Dataset preprocessing and validation

---

Generated: May 18, 2026
