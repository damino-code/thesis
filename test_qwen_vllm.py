"""Quick test: verify vLLM loads a Qwen model and can generate on GPU."""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"
DOWNLOAD_DIR = "/scratch/amine/models"

print(f"Loading model: {MODEL_ID}")
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=1024,
    seed=42,
    trust_remote_code=True,
    download_dir=DOWNLOAD_DIR,
    quantization="awq",
)
print("Model loaded successfully.")

# Use the tokenizer's chat template
tokenizer = llm.get_tokenizer()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the difference between a dilemma and a problem?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(
    max_tokens=150,
    temperature=0.7,
)

print("\nGenerating response...\n")
outputs = llm.generate([prompt], sampling_params)

print("-" * 30)
print(outputs[0].outputs[0].text.strip())
print("-" * 30)
