#!/bin/bash
# Install llama-cpp-python with CUDA support for H200 GPU (compute capability 9.0)

set -e

echo "=========================================="
echo "Installing llama-cpp-python with CUDA support"
echo "=========================================="

# Fix conda CUDA header paths (conda puts them in targets/x86_64-linux/include/)
if [ -d "$CONDA_PREFIX/targets/x86_64-linux/include" ]; then
    echo "Symlinking CUDA headers to \$CONDA_PREFIX/include/..."
    ln -sf $CONDA_PREFIX/targets/x86_64-linux/include/*.h $CONDA_PREFIX/include/ 2>/dev/null || true
fi

# Use conda environment variables for CUDA
export CUDA_PATH=$CONDA_PREFIX
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCUDAToolkit_INCLUDE_DIR=$CONDA_PREFIX/targets/x86_64-linux/include"

pip install llama-cpp-python --force-reinstall --no-cache-dir -v

echo ""
echo "=========================================="
echo "Installation complete! Verifying..."
echo "=========================================="

# Verify CUDA is linked
CUDA_LIBS=$(ldd $(python -c "import llama_cpp; import os; print(os.path.dirname(llama_cpp.__file__))")/lib*.so 2>&1 | grep -i cuda)
if [ -n "$CUDA_LIBS" ]; then
    echo "CUDA support confirmed:"
    echo "$CUDA_LIBS"
else
    echo "WARNING: No CUDA libraries linked. Build may have fallen back to CPU-only."
fi

# Check GPU compute mode
COMPUTE_MODE=$(nvidia-smi -q -d COMPUTE 2>/dev/null | grep "Compute Mode" | awk '{print $NF}')
if [ "$COMPUTE_MODE" = "Prohibited" ]; then
    echo ""
    echo "WARNING: GPU Compute Mode is 'Prohibited' on this node ($(hostname))."
    echo "Try a different node: exit and run:"
    echo "  srun --partition=compute --gres=gpu:nvidia:1 --exclude=$(hostname) --cpus-per-task=8 --mem=64G --time=04:00:00 --pty bash"
fi

python -c "
from llama_cpp import Llama
print('llama-cpp-python imported successfully')
"
