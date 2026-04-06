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

# Debug counter
_debug_count = [0]  # Use list to make it mutable in nested function

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
    log_file = open("server.log", "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    
    print("Waiting for server to become ready (this may take up to 2 minutes)...")
    start_time = time.time()
    while time.time() - start_time < 120:
        time.sleep(2)
        if is_server_running("http://127.0.0.1:8000/v1/models"):
            print("Server is up and running!")
            return process, log_file
            
    print("Error: Server failed to start within the timeout. Check server.log for details.")
    process.terminate()
    log_file.close()
    sys.exit(1)

def parse_llm_response(choice, attribute):
    """Extract score and confidence from the logprobs and text."""
    # For chat completions, content is in choice.message.content
    output = choice.message.content.strip() if hasattr(choice.message, 'content') else str(choice.message).strip()
    
    # Extract number
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        val = float(numbers[0])
        confidence = 0.0
        
        # Calculate Logit-Based Confidence
        if hasattr(choice, 'logprobs') and choice.logprobs and hasattr(choice.logprobs, 'content') and choice.logprobs.content:
            # For chat completions, logprobs are in a different structure
            logprobs_list = []
            for token_data in choice.logprobs.content:
                if hasattr(token_data, 'logprob') and token_data.logprob is not None:
                    logprobs_list.append(token_data.logprob)
            
            if logprobs_list:
                avg_logprob = np.mean(logprobs_list)
                confidence = math.exp(avg_logprob)
                
        return {attribute: val, 'confidence': confidence}
    else:
        print(f"⚠️  No number found in '{attribute}' response: {repr(output)}")
        return {attribute: None, 'confidence': 0.0}

async def fetch_annotation(client, semaphore, user_message, attribute, comment_id, index):
    """Sends a single request to the chat completions endpoint asynchronously."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": user_message}],
                max_tokens=10,
                temperature=0.1,
                stop=["<|eot_id|>"],
                logprobs=1
            )
            
            choice = response.choices[0]
            raw_response = choice.message.content
            
            # Debug: Print prompt and response on first 5 requests globally
            if _debug_count[0] < 5:
                _debug_count[0] += 1
                print(f"\n{'='*100}", flush=True)
                print(f"📤 DEBUG REQUEST #{_debug_count[0]} (Index: {index}, ID: {comment_id}, Attr: {attribute})", flush=True)
                print(f"{'='*100}", flush=True)
                print(f"FULL PROMPT:\n{user_message}", flush=True)
                print(f"\n{'-'*100}", flush=True)
                print(f"📥 RAW RESPONSE:\n'{raw_response}'", flush=True)
                print(f"{'='*100}\n", flush=True)
                sys.stdout.flush()
            
            res = parse_llm_response(choice, attribute)
            res['comment_id'] = comment_id
            res['index'] = index
            return res
        except Exception as e:
            print(f"❌ Error on row {index} (ID: {comment_id}): {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {attribute: None, 'confidence': 0.0, 'comment_id': comment_id, 'index': index}

async def run_async_batch_analysis(attribute_name, sample_size='all', use_dynamic=False):
    print(f"🚀 STARTING ASYNC BATCH ANALYSIS FOR: {attribute_name.upper()}")
    
    try:
        df = load_dataset()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    text_column = get_text_column(df)
    if not text_column:
        print("❌ Could not find text column automatically.")
        return

    # Sample Data
    if str(sample_size).lower() in ['all', 'a']:
        df_sample = df
    else:
        try:
            df_sample = df.sample(n=int(sample_size), random_state=42)
        except:
            df_sample = df
            
    print(f"   Analyzing {len(df_sample)} comments for {attribute_name}...")
    
    # Initialize analyzer WITHOUT a local model instance (passing None)
    analyzer = SingleAttributeAnalyzer(None, use_dynamic=use_dynamic)
    
    # Setup Result Folder
    if use_dynamic:
        attr_result_folder = os.path.join(config.RESULTS_FOLDER, "persona_results", attribute_name)
    else:
        attr_result_folder = os.path.join(config.RESULTS_FOLDER, "single_attribute_analyser", attribute_name)
        
    print(f"📁 Results will be saved to: {attr_result_folder}")
    os.makedirs(attr_result_folder, exist_ok=True)
    
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="dummy_key"
    )
    
    # We limit concurrent requests to avoid OS socket exhaustion. Llama.cpp will queue them internally up to n_batch.
    semaphore = asyncio.Semaphore(150)
    tasks = []
    
    print(f"🏗️ Generating fully formatted prompts locally...")
    for i, (idx, row) in enumerate(df_sample.iterrows(), 1):
        comment = str(row[text_column])
        comment_id = row.get('comment_id', idx) if use_dynamic else None
        
        # Pre-build the prompt (just the user message part, not the full chat formatted prompt)
        try:
            # Get just the user message content (without special tokens)
            if use_dynamic and comment_id is not None:
                user_message = analyzer._build_dynamic_prompt(attribute_name, comment_id, comment)
                if user_message is None:
                    # Fallback to vanilla if dynamic prompt fails
                    user_message = analyzer.prompts[attribute_name].format(text=comment[:500])
            else:
                # Use standard vanilla prompt
                user_message = analyzer.prompts[attribute_name].format(text=comment[:500])
            
            if user_message:
                # Add comment_id fallback
                cid = row.get('comment_id', idx)
                task = fetch_annotation(client, semaphore, user_message, attribute_name, cid, idx)
                tasks.append(task)
            else:
                 print(f"⚠️ Could not build prompt for idx {idx}")
                 cid = row.get('comment_id', idx)
                 tasks.append(asyncio.create_task(
                     asyncio.sleep(0, result={attribute_name: None, 'confidence': 0.0, 'comment_id': cid, 'index': idx})
                 ))
        except Exception as e:
             print(f"⚠️ Error building prompt for idx {idx}: {e}")
             cid = row.get('comment_id', idx)
             tasks.append(asyncio.create_task(
                 asyncio.sleep(0, result={attribute_name: None, 'confidence': 0.0, 'comment_id': cid, 'index': idx})
             ))
             
    print(f"⚡ Firing {len(tasks)} requests asynchronously to the GPU server...")
    start_time = time.time()
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"⏱️ Async inference completed in {end_time - start_time:.2f} seconds.")
    
    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_df = pd.DataFrame(results)
    
    # Ensure comment_id is first column
    if 'comment_id' in out_df.columns:
        cols = list(out_df.columns)
        cols.insert(0, cols.pop(cols.index('comment_id')))
        out_df = out_df[cols]
        
    filename = f"results_{attribute_name}_{timestamp}.csv"
    save_path = os.path.join(attr_result_folder, filename)
    out_df.to_csv(save_path, index=False)
    print(f"✅ Finished {attribute_name}. Saved to: {save_path}")


def main():
    print("🚀 ASYNC BATCH PIPELINE START")
    
    # 1. Download Model (using centralized config from config.py)
    model_path = download_model()
    
    server_process = None
    log_file = None
    try:
        if not is_server_running():
            server_process, log_file = start_server(model_path)
        else:
            print("🟢 Local server already running on port 8000.")
            
        print("\nSelect Analysis Mode:")
        print("1. Vanilla (Standard static prompts)")
        print("2. Feature (Annotator-specific dynamic prompts)")
        
        mode_choice = input("\nEnter choice (1 or 2) [Default: 1]: ").strip()
        use_dynamic = mode_choice == '2'
        
        sample_size = input("\nEnter sample size (or 'all'): ").strip()
        
        if not sample_size:
            sample_size = 'all'
            
        # Running synchronously to keep inputs simple, but using asyncio for each attribute loop
        for attribute in config.ATTRIBUTES:
            print(f"\n{'-'*50}")
            asyncio.run(run_async_batch_analysis(attribute, sample_size=sample_size, use_dynamic=use_dynamic))
            
    finally:
        # Shutdown Background Server if we started it
        if server_process is not None:
            print("\nShutting down background server...")
            server_process.terminate()
            server_process.wait()
            if log_file:
                log_file.close()

if __name__ == "__main__":
    main()
