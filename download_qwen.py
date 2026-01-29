import os
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
# We use the same model folder so all your models are in one place
model_folder = "/scratch/amine/models" 

# Official Qwen 2.5 7B Instruct GGUF repository
repo_id = "Qwen/Qwen2.5-7B-Instruct-GGUF"

# Q4_K_M is the recommended quantization (balanced quality & speed)
filename = "qwen2.5-7b-instruct-q4_k_m.gguf"

# --- DOWNLOAD ---
print(f"Target: {os.path.join(model_folder, filename)}")
print(f"Downloading {filename} from {repo_id}...")

os.makedirs(model_folder, exist_ok=True)

try:
    path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        local_dir=model_folder,
        local_dir_use_symlinks=False # Ensure it's a real file, not a symlink
    )
    print(f"✅ Download complete! File saved to: {path}")
except Exception as e:
    print(f"❌ Download failed: {e}")
