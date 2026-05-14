import os
from vllm import LLM
import config


def load_model(model_id=None, gpu_memory_utilization=None, max_model_len=None,
               tensor_parallel_size=None):
    if model_id is None:
        model_id = config.MODEL_ID
    if gpu_memory_utilization is None:
        gpu_memory_utilization = config.GPU_MEMORY_UTILIZATION
    if max_model_len is None:
        max_model_len = config.MAX_MODEL_LEN
    if tensor_parallel_size is None:
        tensor_parallel_size = config.TENSOR_PARALLEL_SIZE

    print(f"Loading model: {model_id}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print(f"  Max model length: {max_model_len}")
    print(f"  Max concurrent sequences: {config.MAX_NUM_SEQS}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  Seed: {config.SEED}")

    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=config.MAX_NUM_SEQS,
        seed=config.SEED,
        trust_remote_code=True,
        download_dir=config.MODEL_DOWNLOAD_DIR,
        quantization=getattr(config, 'MODEL_QUANTIZATION', 'awq'),
    )

    print("Model loaded successfully.")
    return llm
