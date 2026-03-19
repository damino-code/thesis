import os
import config
from single_attribute_analyser.shared_runner import run_attribute_analysis

def get_all_personas():
    persona_root = os.path.join(os.path.dirname(__file__), "persona_prompts")
    json_files = []
    for root, dirs, files in os.walk(persona_root):
        for f in files:
            if f.endswith(".json"):
                # Get the relative path from persona_root and strip .json
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, persona_root)
                json_files.append(os.path.splitext(rel_path)[0])
    return sorted(json_files)

def main():
    print("🚀 STARTING BATCH ATTRIBUTE ANALYSIS (Qwen)")
    
    # 1. Select Persona
    print("\nAvailable modes:")
    print("0. Vanilla (No Persona)")
    all_personas = get_all_personas()
    for i, p in enumerate(all_personas, 1):
        print(f"{i}. {p}")
        
    choice = input("\nEnter number(s) (e.g. 0,1,5) or 'all': ").strip().lower()
    
    selected_indices = []
    if choice == 'all':
        selected_indices = list(range(len(all_personas) + 1))
    else:
        try:
            selected_indices = [int(x.strip()) for x in choice.split(',')]
        except:
            print("Invalid input.")
            return

    # 2. Sample size
    sample_size = input("Enter sample size (or 'all'): ").strip()
    
    for idx in selected_indices:
        persona_path = None if idx == 0 else all_personas[idx-1]
        p_name = persona_path if persona_path else "VANILLA"
        
        print(f"\n=============================================")
        print(f"RUNNING PROMPT: {p_name}")
        print(f"=============================================")
        
        for attribute in config.ATTRIBUTES:
            print(f"\n   Processing: {attribute.upper()}")
            try:
                run_attribute_analysis(attribute, sample_size=sample_size, persona=persona_path)
            except Exception as e:
                print(f"   ❌ Failed to run {attribute}: {e}")

    print("\n🎉 Batch processing complete.")

if __name__ == "__main__":
    main()
