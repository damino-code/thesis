"""
test_batch.py — Quick async batch test for the llama.cpp server.
Sends N_PARALLEL requests simultaneously to verify the server is working
correctly and to benchmark raw throughput before running the full pipeline.
"""
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

# --- Configuration ---
model_folder = "/scratch/amine/models"
repo_id      = "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF"
filename     = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
model_path   = os.path.join(model_folder, filename)

# H100 VRAM budget:
#   Model (Q4_K_M 8B) : ~4.5 GB
#   KV cache (fp16)   : ~128 KB/token for Llama-3.2-8B (GQA, 8 KV heads)
#   N_PARALLEL=32, 4096 tokens/slot → 131072 total tokens → ~16 GB KV cache
#   Total             : ~21 GB out of 80 GB — very comfortable.
#   To push further: set N_PARALLEL=64, N_CTX=262144 (~39 GB total).
N_PARALLEL = 32
N_CTX      = N_PARALLEL * 4096   # 131072
N_BATCH    = 4096

SYSTEM_MESSAGE = "You are a helpful assistant. Answer briefly and directly."


# --- Model download (only if missing) ---
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Downloading...")
    os.makedirs(model_folder, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_folder)
    print("Download complete!")
else:
    print(f"Model found at {model_path}.")


def is_server_running(url="http://127.0.0.1:8000/v1/models"):
    try:
        response = urllib.request.urlopen(url, timeout=2)
        return response.getcode() == 200
    except (urllib.error.URLError, socket.timeout, ConnectionRefusedError):
        return False


def start_server(model_path):
    print(f"Starting llama_cpp.server (n_ctx={N_CTX}, n_parallel={N_PARALLEL})...")
    cmd = [
        "python", "-m", "llama_cpp.server",
        "--model",      model_path,
        "--n_gpu_layers", "-1",
        "--n_ctx",      str(N_CTX),
        "--n_batch",    str(N_BATCH),
        "--n_parallel", str(N_PARALLEL),
        "--host",       "0.0.0.0",
        "--port",       "8000",
    ]
    log_file = open("server.log", "w")
    process  = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    print("Waiting for server (up to 2 minutes)...")
    start = time.time()
    while time.time() - start < 120:
        time.sleep(2)
        if is_server_running():
            print(f"Server ready — {N_PARALLEL} parallel slots, {N_CTX} context tokens.")
            return process, log_file

    print("Server failed to start. Check server.log.")
    process.terminate()
    log_file.close()
    sys.exit(1)


async def fetch_response(client: AsyncOpenAI, semaphore: asyncio.Semaphore, idx: int, prompt: str):
    """Send one request, return (idx, prompt, response_text, latency_s)."""
    async with semaphore:
        t0 = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=60,
                temperature=0.1,
            )
            text = response.choices[0].message.content
        except Exception as e:
            text = f"ERROR: {e}"
        latency = time.perf_counter() - t0
        return idx, prompt, text, latency


async def main():
    client    = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy_key")
    semaphore = asyncio.Semaphore(N_PARALLEL)  # matches server --n_parallel

    prompts = [
        "Write a short haiku about a sleepy cat.",
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Give a 2-step recipe for scrambled eggs.",
        "Translate 'Hello, how are you?' into Spanish.",
        "Name three planets in our solar system.",
        "Who wrote Romeo and Juliet?",
        "What is the square root of 144?",
        "Summarise The Matrix in two sentences.",
        "Why is the sky blue? One sentence.",
        "What is the boiling point of water in Celsius?",
        "Name the first person to walk on the Moon.",
        "What does CPU stand for?",
        "Convert 100 Fahrenheit to Celsius.",
        "Give one example of a mammal that lives in the ocean.",
        "What year did World War II end?",
        "What language is spoken in Brazil?",
        "What is the chemical symbol for gold?",
        "Who painted the Mona Lisa?",
        "How many continents are there?",
    ]

    print(f"\nSending {len(prompts)} requests with semaphore={N_PARALLEL}...")
    total_start = time.perf_counter()

    tasks   = [fetch_response(client, semaphore, i, p) for i, p in enumerate(prompts, 1)]
    results = await asyncio.gather(*tasks)

    total_elapsed = time.perf_counter() - total_start

    print("\n--- RESULTS ---")
    for idx, prompt, text, latency in sorted(results):
        print(f"\n[{idx:02d}] ({latency:.2f}s) Q: {prompt}")
        print(f"       A: {text.strip()}")

    latencies = [r[3] for r in results]
    print(f"\n--- STATS ---")
    print(f"  Total wall time : {total_elapsed:.2f}s")
    print(f"  Requests        : {len(prompts)}")
    print(f"  Throughput      : {len(prompts)/total_elapsed:.1f} req/s")
    print(f"  Latency p50     : {sorted(latencies)[len(latencies)//2]:.2f}s")
    print(f"  Latency max     : {max(latencies):.2f}s")


if __name__ == "__main__":
    server_process = None
    log_file       = None
    try:
        if not is_server_running():
            server_process, log_file = start_server(model_path)
        else:
            print("Server already running on port 8000.")

        asyncio.run(main())

    finally:
        if server_process is not None:
            print("\nShutting down server...")
            server_process.terminate()
            server_process.wait()
            if log_file:
                log_file.close()
