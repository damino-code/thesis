import config
import os
from single_attribute_analyser.shared_runner import run_attribute_analysis

def main():
    print("🚀 STARTING BATCH ATTRIBUTE ANALYSIS")
    
    # 1. Choose Analysis Mode
    print("\nSelect Analysis Mode:")
    print("1. Vanilla (Standard static prompts)")
    print("2. Feature (Annotator-specific dynamic prompts)")
    
    mode_choice = input("\nEnter choice (1 or 2) [Default: 1]: ").strip()
    use_dynamic = mode_choice == '2'
    
    mode_name = "Feature-based" if use_dynamic else "Vanilla"
    print(f"\n✓ Selected: {mode_name} mode")
    
    # 2. Choose Sample Size
    sample_size = input("\nEnter sample size (or 'all'): ").strip()
    
    # 3. Run Analysis for all attributes
    print(f"\n=============================================")
    print(f"🌟 RUNNING {mode_name.upper()} ANALYSIS")
    print(f"=============================================")
    
    for attribute in config.ATTRIBUTES:
        print(f"\n---------------------------------------------")
        print(f"Processing: {attribute.upper()}")
        print(f"---------------------------------------------")
        try:
            run_attribute_analysis(attribute, sample_size=sample_size, use_dynamic=use_dynamic)
        except Exception as e:
            print(f"❌ Failed to run {attribute}: {e}")

    print("\n✅ All attributes processed.")
    print("📝 Run 'python src/merge_results.py' to combine the files.")

    print("\n🎉 Batch processing complete.")

if __name__ == "__main__":
    main()
