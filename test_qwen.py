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

def choose_filenames(repo_id: str) -> list[str]:
    files = list_repo_files(repo_id)
    gguf_files = [f for f in files if f.lower().endswith(".gguf")]
    if not gguf_files:
        raise RuntimeError("No .gguf files found in repo: " + repo_id)
    
    selected_base = None
    # Try preferred quantizations first
    for quant in preferred:
        for f in gguf_files:
            if quant.lower() in f.lower():
                # If it's a shard, get the base name pattern
                if "-00001-of-" in f:
                    selected_base = f.split("-00001-of-")[0]
                    break
                elif "-of-" not in f:
                    return [f]
        if selected_base:
            break
            
    if selected_base:
        return [f for f in gguf_files if f.startswith(selected_base)]
        
    # Fallback: first gguf file
    return [gguf_files[0]]

filenames = choose_filenames(repo_id)

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
print(f"Selected files: {filenames}")

paths = []
for filename in filenames:
    small_request_test(repo_id, filename)
    target_path = os.path.join(model_folder, filename)
    print(f"Checking {filename}...")

    if is_file_downloaded(model_folder, filename):
        print(f"ℹ️ File already present: {target_path}")
        paths.append(target_path)
    else:
        try:
            print(f"Downloading {filename} from {repo_id}...")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_folder,
            )
            print(f"✅ Download complete! File saved to: {path}")
            paths.append(path)
        except Exception as e:
            print(f"❌ Download failed for {filename}: {e}")

# --- QUICK TEST ---
if paths:
    # Use the first shard/file for the smoke test (llama-cpp handles shards automatically if they share a base name)
    main_path = paths[0]
    try:
        total_size = sum(os.path.getsize(p) for p in paths) / (1024 * 1024)
        print(f"Total model size ({len(paths)} files): {total_size:.2f} MB")
        
        ok = verify_gguf_magic(main_path)
        if ok:
            print(f"✅ GGUF header verified in {os.path.basename(main_path)}.")
        else:
            print(f"⚠️ Could not verify GGUF header in {os.path.basename(main_path)}.")
    except Exception as e:
        print(f"⚠️ Test failed: {e}")
    
    # Run a tiny inference
    run_llama_smoke_test(main_path)
else:
    print("⚠️ No files to test.")
