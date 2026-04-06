import time
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- CONFIGURATION ---
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
    print(f"Model found at {model_path}. Loading into memory...")

# --- STEP 2: LOAD MODEL FOR BATCHING ---
print("Initializing Llama model on GPU...")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,      
    n_ctx=8192,           # Needs to be large enough to fit ALL prompts in the batch simultaneously
    n_batch=1024,
    verbose=False
)

def main():
    prompts = [
        "Write a short, funny haiku about a sleepy cat.",
        "What is the capital of France?",
        "Explain quantum computing in one simple sentence.",
        "Give me a quick recipe for scrambled eggs.",
        "Translate 'Hello, how are you?' into Spanish.",
        "Name three planets in our solar system.",
        "Who wrote the play Romeo and Juliet?",
        "What is the square root of 144?",
        "Write a 2-sentence summary of the movie The Matrix.",
        "Why is the sky blue? Answer in one short sentence."
    ]

    # Format prompts exactly as the LLM expects them for sequential generation
    formatted_prompts = []
    for p in prompts:
        formatted_prompts.append(
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    print(f"Starting direct native batch processing of {len(prompts)} prompts...")
    start_time = time.time()

    # Pass the ENTIRE list directly into the `llm` object
    # This automatically invokes the underlying continuous batching C++ engine!
    outputs = llm(
        formatted_prompts,
        max_tokens=150,
        stop=["<|eot_id|>"],
        echo=False
    )
    
    end_time = time.time()
    
    # Print the results
    print("\n--- RESULTS ---")
    for i, output in enumerate(outputs):
        print(f"\nPrompt: {prompts[i]}\nResponse: {output['choices'][0]['text'].strip()}\n")
        
    print(f"Total time taken (DIRECT BATCH): {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
