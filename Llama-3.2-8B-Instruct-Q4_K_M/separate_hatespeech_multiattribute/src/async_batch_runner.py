import asyncio
import time
import os
import sys
import subprocess
import socket
import urllib.request
import urllib.error
import re
import math
import numpy as np
import pandas as pd
from datetime import datetime
from openai import AsyncOpenAI

# Add parent directory to path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config
from data_loader import load_dataset, get_text_column
from single_attribute_analyzer import SingleAttributeAnalyzer
from model_loader import download_model

# --- Server config ---
# H100 has ~80GB VRAM. Model (Llama-3.2-8B Q4_K_M) uses ~4.5GB.
# KV cache for Llama-3.2-8B (GQA, 8 KV heads, head_dim=128, 32 layers, fp16):
#   ~128KB per token. For 131072 total tokens: ~16GB KV cache.
#   Total VRAM used: ~21GB out of 80GB — well within budget.
# Each parallel slot gets: 131072 / N_PARALLEL = 4096 tokens (ample for ~600-token prompts).
# Increase N_PARALLEL to 64 (n_ctx=262144, ~39GB total) if you want more throughput.
N_PARALLEL = 32
N_CTX      = N_PARALLEL * 4096   # 131072
N_BATCH    = 4096

VANILLA_SYSTEM = "You are an expert content annotator."
DYNAMIC_SYSTEM = (
    "You are an expert content annotator responding from the perspective "
    "defined in the following instructions."
)

# Debug counter
_debug_count = [0]


def is_server_running(url="http://127.0.0.1:8000/v1/models"):
    """Check if the local server is actually ready to receive API requests."""
    try:
        response = urllib.request.urlopen(url, timeout=2)
        return response.getcode() == 200
    except (urllib.error.URLError, socket.timeout, ConnectionRefusedError):
        return False


def start_server(model_path):
    """Launch the llama.cpp server in the background."""
    print(f"Starting llama_cpp.server (n_ctx={N_CTX}, n_batch={N_BATCH}, n_parallel={N_PARALLEL})...")
    cmd = [
        "python", "-m", "llama_cpp.server",
        "--model", model_path,
        "--n_gpu_layers", "-1",
        "--n_ctx",      str(N_CTX),
        "--n_batch",    str(N_BATCH),
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    log_file = open("server.log", "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    print("Waiting for server to become ready (this may take up to 2 minutes)...")
    start_time = time.time()
    while time.time() - start_time < 120:
        time.sleep(2)
        if is_server_running("http://127.0.0.1:8000/v1/models"):
            print(f"Server is up! {N_PARALLEL} parallel slots, {N_CTX} total context.")
            return process, log_file

    print("Error: Server failed to start within the timeout. Check server.log for details.")
    process.terminate()
    log_file.close()
    sys.exit(1)


def parse_llm_response(choice, attribute):
    """Extract score and confidence from the logprobs and text."""
    output = (
        choice.message.content.strip()
        if hasattr(choice.message, "content")
        else str(choice.message).strip()
    )

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        val = float(numbers[0])
        confidence = 0.0

        if (
            hasattr(choice, "logprobs")
            and choice.logprobs
            and hasattr(choice.logprobs, "content")
            and choice.logprobs.content
        ):
            logprobs_list = [
                td.logprob
                for td in choice.logprobs.content
                if hasattr(td, "logprob") and td.logprob is not None
            ]
            if logprobs_list:
                confidence = math.exp(np.mean(logprobs_list))

        return {attribute: val, "confidence": confidence}
    else:
        print(f"⚠️  No number found in '{attribute}' response: {repr(output)}")
        return {attribute: None, "confidence": 0.0}


async def fetch_annotation(client, semaphore, messages, attribute, comment_id, index):
    """Send a single chat request to the llama.cpp server asynchronously.

    Args:
        messages: Full OpenAI messages list, e.g.
                  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="default",
                messages=messages,
                max_tokens=10,
                temperature=0.1,
                logprobs=True,
            )

            choice = response.choices[0]

            # Debug: print first 5 requests
            if _debug_count[0] < 5:
                _debug_count[0] += 1
                print(f"\n{'='*80}", flush=True)
                print(
                    f"DEBUG #{_debug_count[0]} | idx={index} id={comment_id} attr={attribute}",
                    flush=True,
                )
                print(f"SYSTEM: {messages[0]['content'][:120]}", flush=True)
                print(f"USER:   {messages[1]['content'][:200]}", flush=True)
                print(f"RESP:   {repr(choice.message.content)}", flush=True)
                print(f"{'='*80}\n", flush=True)

            res = parse_llm_response(choice, attribute)
            res["comment_id"] = comment_id
            res["index"] = index
            return res

        except Exception as e:
            print(f"❌ Error row={index} id={comment_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {attribute: None, "confidence": 0.0, "comment_id": comment_id, "index": index}


async def run_async_batch_analysis(attribute_name, sample_size="all", use_dynamic=False):
    print(f"\n🚀 ASYNC BATCH ANALYSIS: {attribute_name.upper()}")

    try:
        df = load_dataset()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    text_column = get_text_column(df)
    if not text_column:
        print("❌ Could not find text column automatically.")
        return

    # Sample data
    if str(sample_size).lower() in ["all", "a"]:
        df_sample = df
    else:
        try:
            df_sample = df.sample(n=int(sample_size), random_state=42)
        except Exception:
            df_sample = df

    print(f"   {len(df_sample)} comments × attribute '{attribute_name}'")

    # Analyzer (no local model — only used for prompt building)
    analyzer = SingleAttributeAnalyzer(None, use_dynamic=use_dynamic)

    # Result folder
    sub = "persona_results" if use_dynamic else "single_attribute_analyser"
    attr_result_folder = os.path.join(config.RESULTS_FOLDER, sub, attribute_name)
    os.makedirs(attr_result_folder, exist_ok=True)
    print(f"📁 Results → {attr_result_folder}")

    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy_key")

    # Semaphore matches N_PARALLEL: no more requests in-flight than the server
    # can actually process simultaneously. Avoids OS socket exhaustion too.
    semaphore = asyncio.Semaphore(N_PARALLEL)
    tasks = []

    print(f"🏗️  Building prompts...")
    for _, (idx, row) in enumerate(df_sample.iterrows(), 1):
        comment    = str(row[text_column])
        comment_id = row.get("comment_id", idx)

        try:
            if use_dynamic and comment_id is not None:
                user_message = analyzer._build_dynamic_prompt(attribute_name, comment_id, comment)
                if user_message is None:
                    # Fallback to vanilla
                    user_message   = analyzer.prompts[attribute_name].format(text=comment[:500])
                    system_message = VANILLA_SYSTEM
                else:
                    system_message = DYNAMIC_SYSTEM
            else:
                user_message   = analyzer.prompts[attribute_name].format(text=comment[:500])
                system_message = VANILLA_SYSTEM

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ]
            tasks.append(
                fetch_annotation(client, semaphore, messages, attribute_name, comment_id, idx)
            )

        except Exception as e:
            print(f"⚠️  Prompt build error idx={idx}: {e}")
            # Capture loop variables in default args to avoid closure issues
            async def _null(attr=attribute_name, cid=comment_id, i=idx):
                return {attr: None, "confidence": 0.0, "comment_id": cid, "index": i}
            tasks.append(_null())

    print(f"⚡ Firing {len(tasks)} requests ({N_PARALLEL} in parallel)...")
    start_time = time.time()
    results    = await asyncio.gather(*tasks)
    elapsed    = time.time() - start_time
    print(f"⏱️  Done in {elapsed:.2f}s  ({len(tasks)/elapsed:.1f} req/s)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_df    = pd.DataFrame(results)

    if "comment_id" in out_df.columns:
        cols = list(out_df.columns)
        cols.insert(0, cols.pop(cols.index("comment_id")))
        out_df = out_df[cols]

    filename  = f"results_{attribute_name}_{timestamp}.csv"
    save_path = os.path.join(attr_result_folder, filename)
    out_df.to_csv(save_path, index=False)
    print(f"✅ Saved → {save_path}")


def main():
    print("🚀 ASYNC BATCH PIPELINE START")

    model_path     = download_model()
    server_process = None
    log_file       = None

    try:
        if not is_server_running():
            server_process, log_file = start_server(model_path)
        else:
            print("🟢 Server already running on port 8000.")

        print("\nSelect Analysis Mode:")
        print("1. Vanilla (standard static prompts)")
        print("2. Dynamic (annotator-specific persona prompts)")
        mode_choice = input("\nEnter choice (1 or 2) [Default: 1]: ").strip()
        use_dynamic = mode_choice == "2"

        sample_size = input("\nEnter sample size (or 'all'): ").strip() or "all"

        for attribute in config.ATTRIBUTES:
            print(f"\n{'-'*50}")
            asyncio.run(
                run_async_batch_analysis(attribute, sample_size=sample_size, use_dynamic=use_dynamic)
            )

    finally:
        if server_process is not None:
            print("\nShutting down server...")
            server_process.terminate()
            server_process.wait()
            if log_file:
                log_file.close()


if __name__ == "__main__":
    main()
