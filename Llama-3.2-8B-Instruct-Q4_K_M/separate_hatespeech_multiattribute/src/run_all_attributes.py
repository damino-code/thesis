import config
import os
from single_attribute_analyser.shared_runner import run_attribute_analysis

def main():
    print("🚀 STARTING BATCH ATTRIBUTE ANALYSIS")
    
    # 1. Choose Persona
    persona_base_dir = os.path.join(os.path.dirname(__file__), 'persona_prompts')
    all_personas = []
    if os.path.exists(persona_base_dir):
        # Recursively find all persona directories that contain json files
        for root, dirs, files in os.walk(persona_base_dir):
            if any(f.endswith('.json') for f in files):
                # Get the relative path from persona_base_dir
                rel_path = os.path.relpath(root, persona_base_dir)
                all_personas.append(rel_path)
    
    all_personas.sort()
    print("\nAvailable Personas:")
    print("0. Standard (No Persona)")
    for i, p in enumerate(all_personas, 1):
        # Display the category path for clarity
        display_name = p.replace('_', ' ').title()
        print(f"{i}. {display_name}")
    
    choice = input("\nSelect persona (number(s) or 'all') [Default 0]: ").strip()
    selected_personas = []
    
    if choice.lower() == 'all':
        selected_personas = all_personas
    elif ',' in choice:
        try:
            indices = [int(i.strip()) for i in choice.split(',')]
            selected_personas = [all_personas[i-1] for i in indices if 1 <= i <= len(all_personas)]
        except ValueError:
            print("❌ Invalid multiple selection.")
            return
    elif choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(all_personas):
            selected_personas = [all_personas[idx-1]]
        elif idx == 0:
            selected_personas = [None]
    else:
        selected_personas = [None]

    # 2. Choose Sample Size
    sample_size = input("Enter sample size (or 'all'): ").strip()
    
    # 3. Run Analysis
    for selected_persona in selected_personas:
        persona_display = selected_persona.replace('_', ' ').title() if selected_persona else "Standard"
        print(f"\n=============================================")
        print(f"🌟 PERSONA: {persona_display}")
        print(f"=============================================")
        
        for attribute in config.ATTRIBUTES:
            print(f"\n---------------------------------------------")
            print(f"Processing: {attribute.upper()}")
            print(f"---------------------------------------------")
            try:
                run_attribute_analysis(attribute, sample_size=sample_size, persona=selected_persona)
            except Exception as e:
                print(f"❌ Failed to run {attribute} for {selected_persona}: {e}")

        if selected_persona:
            print(f"\n✅ All attributes processed for {persona_display}.")
            print(f"📝 Run 'python src/merge_results.py \"{selected_persona}\"' to combine the persona files.")
        else:
            print("\n✅ All attributes processed for Standard.")
            print("📝 Run 'python src/merge_results.py' to combine the files.")

    print("\n🎉 Batch processing complete.")

if __name__ == "__main__":
    main()
