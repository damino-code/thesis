import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- CONFIGURATION ---
# Use a folder on your Scratch/Work storage (Replace 'username')
model_folder = "/scratch/amine/models" 
repo_id = "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF"
filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
model_path = os.path.join(model_folder, filename)

# --- STEP 1: DOWNLOAD (Only if missing) ---
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Downloading...")
    os.makedirs(model_folder, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_folder)
    print("Download complete!")
else:
    print(f"Model found at {model_path}. Loading...")

# --- STEP 2: RUN THE MODEL ---
print("Initializing Llama model on GPU...")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,      # -1 = Offload ALL layers to GPU
    n_ctx=2048,           # Small context for a quick test
    verbose=False         # Set to True if you want to see the layer loading logs
)

# --- STEP 3: SEND PROMPT ---
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

what is the difference between a dielemma and a problem ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

print("\nGenerating response...\n")
output = llm(
    prompt,
    max_tokens=100,
    stop=["<|eot_id|>"],
    echo=False
)

print("-" * 30)
print(output['choices'][0]['text'])
print("-" * 30)
