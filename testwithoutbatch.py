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

# --- STEP 2: LOAD MODEL ---
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=8192,
    verbose=False
)

def fetch_response(prompt: str):
    """Sends a single request to the local model sequentially."""
    print(f"Processing: {prompt}")
    # Use the chat_format directly so we don't have to manually insert Llama-3 tags
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return prompt, response['choices'][0]['message']['content']
    except Exception as e:
        return prompt, f"Error: {e}"

def main():
    # A list of 10 varied text prompts
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

    print(f"Starting sequential processing of {len(prompts)} prompts (Direct Python Model)...")
    start_time = time.time()

    results = []
    # Send requests sequentially, ONE AT A TIME, waiting for each to finish
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Sending prompt...")
        result = fetch_response(prompt)
        results.append(result)

    end_time = time.time()
    
    # Print the results
    print("\n--- RESULTS ---")
    for prompt, result in results:
        print(f"\nPrompt: {prompt}\nResponse: {result}\n")
        
    print(f"Total time taken (SEQUENTIAL DIRECT): {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()