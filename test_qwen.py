import os
import time
import requests
from huggingface_hub import hf_hub_download, list_repo_files, hf_hub_url

# --- CONFIGURATION ---
# Pick a writable model directory with fallbacks; override via MODEL_DIR.

def pick_model_dir():
    candidates = []
    env_dir = os.environ.get("MODEL_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.append("/scratch/amine/models")
    candidates.append(os.path.join(os.path.expanduser("~"), "models"))
    candidates.append(os.path.join(os.getcwd(), "models"))
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            continue
    raise RuntimeError("No writable model directory found. Set MODEL_DIR to a writable path.")

model_folder = pick_model_dir()

# Official Qwen 2.5 7B Instruct GGUF repository
repo_id = "Qwen/Qwen2.5-7B-Instruct-GGUF"

# Desired quantization preference order
preferred = ["Q4_K_M", "Q4_K_S", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_0"]

def choose_filename(repo_id: str) -> str:
    files = list_repo_files(repo_id)
    gguf_files = [f for f in files if f.lower().endswith(".gguf")]
    if not gguf_files:
        raise RuntimeError("No .gguf files found in repo: " + repo_id)
    # Try preferred quantizations first
    for quant in preferred:
        for f in gguf_files:
            if quant.lower() in f.lower():
                return f
    # Fallback: first gguf file
    return gguf_files[0]

filename = choose_filename(repo_id)

def small_request_test(repo_id: str, filename: str | None = None) -> None:
    try:
        t0 = time.time()
        r = requests.get(f"https://huggingface.co/api/models/{repo_id}", timeout=10)
        dt = (time.time() - t0) * 1000
        print(f"Network test: repo API status {r.status_code} in {dt:.0f} ms")
    except Exception as e:
        print(f"Network test: repo API request failed: {e}")

    if filename:
        try:
            url = hf_hub_url(repo_id=repo_id, filename=filename)
            t0 = time.time()
            h = requests.head(url, timeout=10, allow_redirects=True)
            dt = (time.time() - t0) * 1000
            print(f"Network test: file HEAD {h.status_code} in {dt:.0f} ms")
        except Exception as e:
            print(f"Network test: file HEAD failed: {e}")

def is_file_downloaded(folder: str, fname: str) -> bool:
    target = os.path.join(folder, fname)
    return os.path.isfile(target) and os.path.getsize(target) > 0

def verify_gguf_magic(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == b"GGUF"
    except Exception:
        return False

def run_llama_smoke_test(model_path: str) -> None:
    try:
        from llama_cpp import Llama
    except Exception as e:
        print(f"⚠️ Llama test skipped (import failed): {e}")
        return

    try:
        print("Initializing Llama model (CPU test)...")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
        prompt = "You are a helpful assistant. Answer briefly: What is the difference between a problem and a dielemma?"
        print("Generating a short response...")
        out = llm(prompt, max_tokens=100, echo=False)
        text = out.get("choices", [{}])[0].get("text", "")
        print("--- Llama test output ---")
        print(text.strip())
        print("-------------------------")
    except Exception as e:
        print(f"⚠️ Llama test failed: {e}")

# --- DOWNLOAD ---
print(f"Target directory selected: {model_folder}")
print(f"Selected file: {filename}")
small_request_test(repo_id, filename)
target_path = os.path.join(model_folder, filename)
print(f"Downloading from {repo_id}...")

if is_file_downloaded(model_folder, filename):
    print(f"ℹ️ File already present: {target_path}")
    path = target_path
else:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_folder,
        )
        print(f"✅ Download complete! File saved to: {path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        path = None

# --- QUICK TEST ---
if path and os.path.isfile(path):
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        ok = verify_gguf_magic(path)
        if ok:
            print("✅ GGUF header verified (magic 'GGUF' detected).")
        else:
            print("⚠️ Could not verify GGUF header; file may be incomplete or not GGUF.")
    except Exception as e:
        print(f"⚠️ Test failed: {e}")
    # Run a tiny inference to confirm the model loads
    run_llama_smoke_test(path)
else:
    print("⚠️ No file to test.")
