import os
import sys
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import config

def download_model(model_folder=config.MODEL_FOLDER, repo_id=config.REPO_ID, filename=config.FILENAME):
    model_path = os.path.join(model_folder, filename)
    print(f"Target Model Path: {model_path}")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")
        os.makedirs(model_folder, exist_ok=True)
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_folder)
        print("Download complete!")
    else:
        print(f"✅ Model found at {model_path}. Ready to load.")
    
    return model_path

def load_model(model_path=None, n_gpu_layers=-1, n_ctx=8192, verbose=False):
    if model_path is None:
        model_path = config.MODEL_PATH

    print("Loading model into RAM...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=verbose,
        logits_all=True  # Required for logprobs/confidence calculation
    )
    print("✅ Model loaded successfully.")
    return llm
