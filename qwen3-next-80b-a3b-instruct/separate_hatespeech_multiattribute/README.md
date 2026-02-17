# Single Attribute Hatespeech Analysis (OpenRouter)

This project performs independent analysis for each attribute using an OpenRouter-hosted model and specific, isolated prompts. The structure and workflow mirror the Llama-based version, but are adapted for OpenRouter.

## Project Structure

```
separate_hatespeech_multiattribute/
├── results/              # Output folder
│   ├── results_sentiment_YYYY...csv   # Individual result files
│   ├── results_insult_YYYY...csv
│   └── combined_results_YYYY...csv    # Master merged file
└── src/
    ├── config.py                     # Configuration paths
    ├── single_attribute_analyzer.py  # Contains the specific prompts per attribute
    ├── run_analysis.py               # Main script: runs analysis & merges results
    ├── evaluation.py                 # Evaluates results against human annotations
   ├── model_loader.py               # OpenRouter client
    └── data_loader.py                # Loads dataset
```

## How to Run

1. **Install Dependencies**
   Ensure you have the requirements installed (same as the main thesis project).

2. **Run Analysis**
   ```bash
   python src/run_analysis.py
   ```
   - You will be asked to select which attributes to run (e.g., `sentiment, insult` or `all`).
   - The script will generate one CSV file per attribute.
   - **At the end**, it automatically merges all generated files into a single `combined_results_....csv`.

3. **Evaluate Results**
   ```bash
   python src/evaluation.py
   ```
   - This script automatically finds the most recent result file for each attribute in the `results/` folder.
   - It compares the predictions against the original `selected_comments.csv` human annotations.
   - It prints Correlation scores (for numeric scales) and Accuracy (for Hate Speech).

## Configuration

Modify `src/config.py` to change:
- `OPENROUTER_MODEL`: The OpenRouter model slug (set via env or in config).
- `DATASET_PATH`: Location of your input CSV.
- `ATTRIBUTES`: List of available attributes to analyze.

## OpenRouter Setup

Set your API key in the environment:

```bash
export OPENROUTER_API_KEY="YOUR_KEY"
export OPENROUTER_MODEL="qwen/qwen3-next-80b-a3b-instruct"
```
