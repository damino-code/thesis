import asyncio
import time
import os
import subprocess
import socket
import sys
import urllib.request
import urllib.error
from openai import AsyncOpenAI
from huggingface_hub import hf_hub_download

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
    print(f"Model found at {model_path}.")

# --- STEP 2: ENSURE SERVER IS RUNNING ---
def is_server_running(url="http://127.0.0.1:8000/v1/models"):
    """Check if the local server is actually ready to receive API requests."""
    try:
        response = urllib.request.urlopen(url, timeout=2)
        return response.getcode() == 200
    except (urllib.error.URLError, socket.timeout, ConnectionRefusedError):
        return False

def start_server(model_path):
    """Launch the llama.cpp server in the background."""
    print("Starting llama_cpp.server in the background (logging to server.log)...")
    cmd = [
        "python", "-m", "llama_cpp.server",
        "--model", model_path,
        "--n_gpu_layers", "-1",
        "--n_ctx", "65536",
        "--n_batch", "4096",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    # Keep the process open and log to a file instead of console
    log_file = open("server.log", "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    
    print("Waiting for server to become ready (this may take up to 2 minutes)...")
    start_time = time.time()
    while time.time() - start_time < 120:
        # Give it a second between polls so we don't spam
        time.sleep(2)
        if is_server_running("http://127.0.0.1:8000/v1/models"):
            print("Server is up and running!")
            return process, log_file
            
    print("Error: Server failed to start within the timeout. Check server.log for details.")
    process.terminate()
    log_file.close()
    sys.exit(1)


async def fetch_response(client: AsyncOpenAI, prompt: str):
    """Sends a single request to the local server."""
    try:
        response = await client.chat.completions.create(
            model="default",  # Model name doesn't typically matter for local llama.cpp
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return prompt, response.choices[0].message.content
    except Exception as e:
        return prompt, f"Error: {e}"

async def main():
    # Initialize the async client pointing to the local llama.cpp server
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy_key"
    )

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

    print(f"Starting batch processing of {len(prompts)} prompts...")
    start_time = time.time()

    # Create a list of tasks for the event loop
    tasks = [fetch_response(client, prompt) for prompt in prompts]

    # Run all tasks concurrently and wait for them to finish
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    
    # Print the results
    print("\n--- RESULTS ---")
    for prompt, result in results:
        print(f"\nPrompt: {prompt}\nResponse: {result}\n")
        
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    server_process = None
    log_file = None
    try:
        if not is_server_running("http://127.0.0.1:8000/v1/models"):
            server_process, log_file = start_server(model_path)
        else:
            print("Local server is already running on port 8000.")
            
        asyncio.run(main())
        
    finally:
        # Step 3: Cleanup if we were the ones to start the background server
        if server_process is not None:
            print("\nShutting down the background server process...")
            server_process.terminate()
            server_process.wait()
            if log_file:
                log_file.close()
