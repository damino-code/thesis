import config

from single_attribute_analyser.shared_runner import run_attribute_analysis

def main():
    print("🚀 STARTING BATCH ATTRIBUTE ANALYSIS (OpenRouter)")
    
    # Ask for sample size once
    sample_size = input("Enter sample size (or 'all'): ").strip()
    
    for attribute in config.ATTRIBUTES:
        print(f"\n---------------------------------------------")
        print(f"Processing: {attribute.upper()}")
        print(f"---------------------------------------------")
        try:
            run_attribute_analysis(attribute, sample_size=sample_size)
        except Exception as e:
            print(f"❌ Failed to run {attribute}: {e}")

    print("\n🎉 All attributes processed.")
    print("Run 'python src/merge_results.py' to combine the files.")

if __name__ == "__main__":
    main()
