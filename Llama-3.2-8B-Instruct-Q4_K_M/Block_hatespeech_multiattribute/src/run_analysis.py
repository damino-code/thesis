import time
import os
import pandas as pd
from datetime import datetime
import config
from model_loader import download_model, load_model
from data_loader import load_dataset, get_text_column
from feature_analyzer import FeatureAnalyzer
from fewshot_analyzer import FewShotAnalyzer

def main():
    # 1. Setup
    print("🚀 AUTOMATED SETUP FOR MULTI-PROFILE ANALYSIS")
    model_path = download_model()
    llm = load_model(model_path)
    
    try:
        df = load_dataset()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Configure Analysis
    text_column = get_text_column(df)
    if not text_column:
        print(f"❌ No text column found. Available columns: {list(df.columns)}")
        text_column = input("Enter column name: ").strip()
    
    print(f"✅ Using text column: '{text_column}'")

    # Sample Size
    total_comments = len(df)
    print(f"\n📈 Total comments in dataset: {total_comments}")
    sample_size_input = input(f"Enter sample size (or 'all' for full dataset): ").strip()
    
    if sample_size_input.lower() in ['all', 'a']:
        df_sample = df
    else:
        try:
            sample_size = int(sample_size_input)
            if sample_size < total_comments:
                df_sample = df.sample(n=sample_size, random_state=42)
            else:
                df_sample = df
        except ValueError:
            print("Invalid input, using full dataset.")
            df_sample = df

    print(f"\n🎯 Will analyze {len(df_sample)} comments")

    # 3. Choose Analyzer
    print("\n🤔 Choose Analysis Type:")
    print("  1. Feature Analysis (Standard Prompting)")
    print("  2. Few-Shot Analysis (With Examples)")
    
    analysis_choice = input("Enter choice (1 or 2): ").strip()
    
    if analysis_choice == '2':
        print("🚀 Starting Few-Shot Analysis")
        analyzer = FewShotAnalyzer(llm)
        file_prefix = "fewshot_analysis"
        
        # Single Pass for Few-Shot
        results = []
        total = len(df_sample)
        start_time = time.time()
        
        print(f"\n📊 Processing {total} comments...")
        
        for i, (idx, row) in enumerate(df_sample.iterrows(), 1):
            try:
                comment_text = str(row[text_column])
                if len(comment_text.strip()) == 0:
                    continue

                result = analyzer.analyze(comment_text)

                # Add metadata
                result['comment_id'] = row.get('comment_id', idx)
                result['index'] = idx
                
                results.append(result)

                if i % 10 == 0 or i == total:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    remaining = (total - i) * avg_time
                    avg_hs = sum(r['hatespeech'] for r in results) / len(results)
                    print(f"[{i}/{total}] ({i/total*100:.1f}%) | Avg HS: {avg_hs:.2f} | ETA: {remaining:.1f}s")
                    
            except Exception as e:
                print(f"⚠️  Error processing comment {i}: {e}")
                continue
                
        # Save Few-Shot Results
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{file_prefix}_{timestamp}.csv"
        output_path = os.path.join(config.RESULTS_FOLDER, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")

    else:
        print("🚀 Starting Multi-Profile Feature Analysis")
        analyzer = FeatureAnalyzer(llm)
        
        # Profile Selection Logic
        print("\n🎭 Available Demographic Profiles:")
        for i, (name, details) in enumerate(analyzer.demographic_profiles.items(), 1):
            print(f"{i}. {name} - {details['description']}")
            
        print("\nSelect profiles (e.g., '1,3,5' or 'all'):")
        profile_input = input("Selection: ").strip().lower()
        
        if profile_input == 'all':
            selected_profiles = list(analyzer.demographic_profiles.keys())
        else:
            try:
                indices = [int(x.strip()) - 1 for x in profile_input.split(',')]
                all_profiles = list(analyzer.demographic_profiles.keys())
                selected_profiles = [all_profiles[i] for i in indices if 0 <= i < len(all_profiles)]
            except:
                print("⚠️ Invalid selection, defaulting to 'neutral_baseline'")
                selected_profiles = ['neutral_baseline']
        
        if not selected_profiles:
             selected_profiles = ['neutral_baseline']
             
        print(f"✅ Selected profiles: {selected_profiles}")

        # Multi-stage loop: Profile -> Data
        all_results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for profile_idx, profile_name in enumerate(selected_profiles, 1):
            print(f"\n{'='*60}")
            print(f"📊 ANALYZING WITH PROFILE {profile_idx}/{len(selected_profiles)}: {profile_name.upper()}")
            print(f"{'='*60}")
            
            profile_results = []
            start_time = time.time()
            total = len(df_sample)
            
            for i, (idx, row) in enumerate(df_sample.iterrows(), 1):
                try:
                    comment_text = str(row[text_column])
                    if len(comment_text.strip()) == 0:
                        continue

                    # Analyze with specific profile
                    result = analyzer.analyze_with_profile(comment_text, profile_name)

                    # Add metadata
                    result['comment_id'] = row.get('comment_id', idx)
                    result['index'] = idx
                    
                    profile_results.append(result)

                    if i % 10 == 0 or i == total:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / i
                        remaining = (total - i) * avg_time
                        avg_hs = sum(r['hatespeech'] for r in profile_results) / len(profile_results)
                        print(f"[{i}/{total}] | HS: {avg_hs:.2f} | ETA: {remaining:.1f}s")

                except Exception as e:
                    print(f"⚠️  Error processing comment {i}: {e}")
                    continue
            
            # Save individual profile results immediately
            if profile_results:
                profile_df = pd.DataFrame(profile_results)
                profile_filename = f"profile_{profile_name}_{timestamp}.csv"
                profile_path = os.path.join(config.RESULTS_FOLDER, profile_filename)
                profile_df.to_csv(profile_path, index=False)
                print(f"\n💾 Saved profile data to: {profile_filename}")
                
                all_results.extend(profile_results)

        # Save Combined Results
        if all_results:
            combined_df = pd.DataFrame(all_results)
            combined_filename = f"multi_attribute_analysis_COMBINED_{timestamp}.csv"
            combined_path = os.path.join(config.RESULTS_FOLDER, combined_filename)
            combined_df.to_csv(combined_path, index=False)
            print(f"\n💾 Final Combined Results saved to: {combined_path}")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
