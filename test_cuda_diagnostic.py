#!/usr/bin/env python
"""Quick CUDA diagnostic script"""
import sys
import os

print("=" * 70)
print("CUDA DIAGNOSTIC CHECK")
print("=" * 70)

# 1. Check llama-cpp-python
try:
    from llama_cpp import Llama
    print("✓ llama-cpp-python imported successfully")
except ImportError as e:
    print(f"✗ llama-cpp-python import failed: {e}")
    sys.exit(1)

# 2. Check for CUDA support in llama-cpp
try:
    import llama_cpp
    # Try to check if built with CUDA
    result = Llama(model_path="/nonexistent", n_gpu_layers=-1)
except Exception as e:
    error_msg = str(e)
    if "CUDA" in error_msg or "cuda" in error_msg:
        print(f"✗ CUDA Error: {error_msg}")
    elif "No such file" in error_msg or "cannot open" in error_msg:
        print("✓ llama-cpp can initialize (file error is expected)")
    else:
        print(f"? Unexpected error: {error_msg}")

# 3. Try to import torch if available
try:
    import torch
    print(f"✓ PyTorch available: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("⚠ PyTorch not installed (optional)")

# 4. Check system CUDA
os.system("echo '---'")
os.system("echo 'System CUDA info:'")
os.system("nvcc --version 2>&1 | head -3")
os.system("nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader")

print("=" * 70)
