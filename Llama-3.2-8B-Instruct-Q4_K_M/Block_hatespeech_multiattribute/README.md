# Hatespeech Analysis Project

This project contains Python scripts converted from `hatespeech_analysis.ipynb`.

## Structure

- `src/`: Contains all source code.
  - `config.py`: Configuration paths and settings.
  - `run_analysis.py`: Main script to run the LLM analysis.
  - `evaluation.py`: Script to evaluate results against human annotations.
  - `model_loader.py`: Handles model downloading and loading.
  - `data_loader.py`: Handles dataset loading.
  - `analyzer.py`: The core `MultiAttributeAnalyzer` class.
  - `visualization.py`: Plotting functions.
- `results/`: Stores output CSVs and visualization plots.

## How to Run

1. **Install Dependencies**
   Ensure you have the requirements installed:
   ```bash
   pip install -r ../requirements.txt
   ```
   (Or use `mamba`/`conda` as per your environment)

2. **Run Analysis**
   To start the analysis (dataset loading -> model inference -> saving results):
   ```bash
   python src/run_analysis.py
   ```
   Results will be saved in `results/multi_attribute_analysis_YYYYMMDD_HHMMSS.csv`.

3. **Run Evaluation**
   To evaluate the generated results against human annotations:
   ```bash
   python src/evaluation.py
   ```
   This will look for the most recent result CSV in `results/` and compare it with the dataset.
   Metrics and plots will be saved in `results/` and `results/visualizations/`.

## Configuration

Edit `src/config.py` to change:
- Model paths
- Dataset location
- Output folders
