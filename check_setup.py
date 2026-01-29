import sys
import os
import shutil

print("=" * 80)
print("🚀 SYSTEM & ENVIRONMENT CHECK")
print("=" * 80)

# 1. Check Python Libraries
print("\n📚 Checking Libraries:")
print(f"   Python Version: {sys.version.split()[0]}")
required_libs = [
    'llama_cpp', 
    'huggingface_hub', 
    'pandas', 
    'numpy', 
    'sklearn', 
    'matplotlib', 
    'seaborn',
    'tqdm'
]

all_libs_ok = True
for lib in required_libs:
    try:
        __import__(lib)
        print(f"  ✅ {lib:<20} Found")
    except ImportError as e:
        print(f"  ❌ {lib:<20} MISSING")
        all_libs_ok = False

if not all_libs_ok:
    print("\n  ⚠️  Please install missing packages:")
    print("      mamba install -c conda-forge pandas numpy scikit-learn matplotlib seaborn huggingface_hub")
    print("      pip install llama-cpp-python")

# 2. Check GPU
print("\n🖥️  Checking GPU:")
if shutil.which('nvidia-smi'):
    exit_code = os.system("nvidia-smi > /dev/null 2>&1")
    if exit_code == 0:
        print("  ✅ NVIDIA GPU detected (nvidia-smi available)")
    else:
        print("  ⚠️  nvidia-smi found but returned error")
else:
    print("  ❌ No NVIDIA GPU detected (nvidia-smi command missing)")
    print("      Inference will be VERY SLOW on CPU.")

# 3. Check Dataset
print("\n📂 Checking Dataset:")
dataset_path = os.path.join(os.getcwd(), "Dataset", "selected_comments.csv")
if os.path.exists(dataset_path):
    print(f"  ✅ Dataset found: {dataset_path}")
else:
    print(f"  ❌ Dataset NOT found at: {dataset_path}")
    print("      Please ensure 'Dataset/' folder exists in this directory.")

# 4. Check Model
print("\n🤖 Checking Model:")
# Using the path defined in your configs
model_folder = "/scratch/amine/models"
model_filename = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
model_path = os.path.join(model_folder, model_filename)

if os.path.exists(model_path):
    print(f"  ✅ Model found: {model_path}")
else:
    print(f"  ❌ Model NOT found at: {model_path}")
    print("      It will be downloaded automatically when you run the analysis script.")

print("\n" + "=" * 80)
if all_libs_ok:
    print("✅ System looks ready!")
else:
    print("❌ Checking failed. Please fix identifiers above.")
