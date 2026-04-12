#!/bin/bash
# Install llama-cpp-python with CUDA support for H200 GPU (compute capability 9.0)

set -e

echo "=========================================="
echo "Installing llama-cpp-python with CUDA support"
echo "=========================================="

# Use conda environment variables for CUDA
export CUDA_PATH=$CONDA_PREFIX
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install with CUDA support enabled
# CMAKE_CUDA_ARCHITECTURES=90 is for H200 GPU (compute capability 9.0)
pip install llama-cpp-python --force-reinstall --no-cache-dir \
  --config-settings='cmake.args=-DGGML_CUDA=ON;-DCMAKE_CUDA_ARCHITECTURES=90' \
  -v

echo ""
echo "=========================================="
echo "Installation complete!"
echo "Verifying CUDA support..."
echo "=========================================="

python -c "
import llama_cpp
print('✓ llama-cpp-python version:', llama_cpp.__version__)
try:
    from llama_cpp import Llama
    print('✓ Can import Llama class')
    # Try to see if CUDA backend is available
    import llama_cpp.llama_cpp as cpp
    if hasattr(cpp, 'LLAMA_CUDA'):
        print('✓ CUDA support detected')
    else:
        print('⚠ CUDA support may not be detected in bindings, but could still work')
except Exception as e:
    print('✗ Error:', e)
"
