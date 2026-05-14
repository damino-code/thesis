import config
from single_attribute_analyser.shared_runner import run_attribute_analysis
from model_loader import load_model


def main():
    print("STARTING BATCH ATTRIBUTE ANALYSIS (vLLM)")

    # 1. Choose Analysis Mode
    print("\nSelect Analysis Mode:")
    print("1. Vanilla (Standard static prompts)")
    print("2. Feature (Annotator-specific dynamic prompts)")

    mode_choice = input("\nEnter choice (1 or 2) [Default: 1]: ").strip()
    use_dynamic = mode_choice == '2'
    mode_name = "Feature-based" if use_dynamic else "Vanilla"
    print(f"\nSelected: {mode_name} mode")

    # 2. Choose Sample Size
    sample_size = input("\nEnter sample size (or 'all'): ").strip()

    # 3. Load model ONCE
    print("\nLoading model (once for all attributes)...")
    llm = load_model()
    print("Model ready.\n")

    # 4. Run Analysis for all attributes
    print(f"=============================================")
    print(f"RUNNING {mode_name.upper()} ANALYSIS")
    print(f"=============================================")

    for attribute in config.ATTRIBUTES:
        print(f"\n---------------------------------------------")
        print(f"Processing: {attribute.upper()}")
        print(f"---------------------------------------------")
        try:
            run_attribute_analysis(
                attribute,
                sample_size=sample_size,
                use_dynamic=use_dynamic,
                llm=llm,
            )
        except Exception as e:
            print(f"Failed to run {attribute}: {e}")

    print("\nAll attributes processed.")
    print("Run 'python src/merge_results.py' to combine the files.")


if __name__ == "__main__":
    main()
