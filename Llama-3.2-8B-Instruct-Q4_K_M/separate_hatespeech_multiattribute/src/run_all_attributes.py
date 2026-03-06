import config
import os
from single_attribute_analyser.shared_runner import run_attribute_analysis

def main():
    print("🚀 STARTING BATCH ATTRIBUTE ANALYSIS")
    
    # 1. Choose Persona
    persona_base_dir = os.path.join(os.path.dirname(__file__), 'persona_prompts')
    personas = []
    if os.path.exists(persona_base_dir):
        personas = [d for d in os.listdir(persona_base_dir) if os.path.isdir(os.path.join(persona_base_dir, d))]
    
    print("\nAvailable Personas:")
    print("0. Standard (No Persona)")
    for i, p in enumerate(personas, 1):
        print(f"{i}. {p.replace('_', ' ').title()}")
    
    choice = input("\nSelect persona (number) [Default 0]: ").strip()
    selected_persona = None
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(personas):
            selected_persona = personas[idx-1]
        elif idx == 0:
            selected_persona = None
    
    # 2. Choose Sample Size
    sample_size = input("Enter sample size (or 'all'): ").strip()
    
    # 3. Run Analysis
    for attribute in config.ATTRIBUTES:
        print(f"\n---------------------------------------------")
        print(f"Processing: {attribute.upper()}")
        print(f"---------------------------------------------")
        try:
            run_attribute_analysis(attribute, sample_size=sample_size, persona=selected_persona)
        except Exception as e:
            print(f"❌ Failed to run {attribute}: {e}")

    print("\n🎉 All attributes processed.")
    if selected_persona:
        print(f"Run 'python src/merge_results.py {selected_persona}' to combine the persona files.")
    else:
        print("Run 'python src/merge_results.py' to combine the files.")

if __name__ == "__main__":
    main()
