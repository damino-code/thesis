import pandas as pd
import numpy as np

# Select 200 comments with highest variance from processed_dataset.csv

def load_processed_dataset():
    """Load the processed dataset"""
    dataset_path = 'Dataset/processed_dataset.csv'
    print(f"📚 Loading {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Loaded dataset with {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"❌ {dataset_path} not found!")
        print("Make sure the file exists in the Dataset directory.")
        return None

def calculate_variance_score(df):
    """Calculate variance score for each comment across annotator demographic columns"""
    print("📊 Calculating variance scores across annotator demographics...")
    
    # Annotator demographic columns to calculate variance across
    variance_columns = [
        'annotator_gender', 'annotator_ideology', 'annotator_age', 
        'annotator_race', 'annotator_religion', 'annotator_sexuality'
    ]
    
    # Check which columns exist in the dataset
    existing_columns = [col for col in variance_columns if col in df.columns]
    missing_columns = [col for col in variance_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️  Missing columns: {missing_columns}")
    
    if not existing_columns:
        print("❌ No annotator demographic columns found!")
        return df
    
    print(f"✅ Using columns for variance: {existing_columns}")
    
    # For categorical/string columns, we need to convert to numeric first
    df_numeric = df.copy()
    
    for col in existing_columns:
        if df[col].dtype == 'object':
            # Convert categorical to numeric codes
            df_numeric[col + '_numeric'] = pd.Categorical(df[col]).codes
            existing_columns[existing_columns.index(col)] = col + '_numeric'
    
    # Calculate variance for each comment across annotator demographic columns
    variance_scores = df_numeric[existing_columns].var(axis=1)
    df['variance_score'] = variance_scores
    
    print(f"✅ Variance scores calculated. Range: {variance_scores.min():.3f} to {variance_scores.max():.3f}")
    return df

def select_high_variance_comments(df, n=200):
    """Select n comments with highest variance, ensuring no duplicate comment_ids"""
    print(f"\n🎯 Selecting {n} comments with highest variance...")
    
    # First, remove duplicates by comment_id, keeping the one with highest variance
    print(f"📋 Original dataset: {len(df)} rows")
    
    # Group by comment_id and keep the row with highest variance for each comment
    df_unique = df.loc[df.groupby('comment_id')['variance_score'].idxmax()]
    print(f"📊 Unique comments: {len(df_unique)} (after removing duplicate IDs)")
    
    # Sort by variance score (highest first) and select top n
    df_sorted = df_unique.sort_values('variance_score', ascending=False)
    
    if len(df_sorted) < n:
        print(f"⚠️  Only {len(df_sorted)} unique comments available, using all")
        selected = df_sorted
    else:
        selected = df_sorted.head(n)
    
    print(f"✅ Selected {len(selected)} comments")
    print(f"📈 Variance range in selection: {selected['variance_score'].min():.3f} to {selected['variance_score'].max():.3f}")
    
    return selected

def main():
    # Load processed dataset
    df = load_processed_dataset()
    if df is None:
        return
    
    print(f"📊 Dataset columns: {len(df.columns)} total columns")
    print(f"📋 Dataset shape: {df.shape}")

    # Calculate variance scores for each comment
    df = calculate_variance_score(df)
    
    # Add variance statistics for context across ALL comments
    print(f"\n📈 VARIANCE STATISTICS ACROSS ALL COMMENTS:")
    print(f"   Total unique comments: {df['comment_id'].nunique()}")
    print(f"   Variance range: {df['variance_score'].min():.3f} to {df['variance_score'].max():.3f}")
    print(f"   Mean variance: {df['variance_score'].mean():.3f}")
    print(f"   Median variance: {df['variance_score'].median():.3f}")
    print(f"   75th percentile: {df['variance_score'].quantile(0.75):.3f}")
    print(f"   90th percentile: {df['variance_score'].quantile(0.90):.3f}")
    print(f"   95th percentile: {df['variance_score'].quantile(0.95):.3f}")
    
    # Select 200 comments with highest variance (no duplicate IDs)
    sample = select_high_variance_comments(df, n=200)
    
    # Include ALL columns from the original dataset
    result = sample.copy()
    
    # Add variance context columns
    all_variances = df.groupby('comment_id')['variance_score'].max()  # Get max variance per comment
    result['variance_percentile'] = all_variances.rank(pct=True) * 100
    result['variance_rank'] = all_variances.rank(ascending=False, method='min')
    
    # Map percentiles and ranks to our selected comments
    result['variance_percentile'] = result['comment_id'].map(
        lambda x: all_variances[all_variances.index == x].rank(pct=True).iloc[0] * 100 if x in all_variances.index else 0
    )
    result['variance_rank'] = result['comment_id'].map(
        lambda x: all_variances[all_variances.index == x].rank(ascending=False, method='min').iloc[0] if x in all_variances.index else 0
    )
    
    # Reorder columns to put key info first
    key_columns = ['comment_id', 'variance_score', 'variance_percentile', 'variance_rank', 'text']
    other_columns = [col for col in result.columns if col not in key_columns and col in df.columns]
    result = result[key_columns + other_columns]

    # Print summary and save to file
    print(f"\n📋 SELECTION SUMMARY:")
    print(f"Total comments selected: {len(result)}")
    print(f"Unique comment IDs: {result['comment_id'].nunique()}")
    print(f"Variance score range in selection: {result['variance_score'].min():.3f} to {result['variance_score'].max():.3f}")
    print(f"Average variance score: {result['variance_score'].mean():.3f}")
    print(f"Selected comments represent the top demographic diversity in the dataset")
    
    # Show top 5 highest variance comments with context
    print(f"\n🔥 TOP 5 HIGHEST VARIANCE COMMENTS (Most Demographically Diverse):")
    print("Rank | ID      | Variance | Text Preview")
    print("-" * 70)
    for i, row in result.head(5).iterrows():
        text_preview = row['text'][:45] + "..." if len(row['text']) > 45 else row['text']
        rank = int(row['variance_rank']) if not pd.isna(row['variance_rank']) else 'N/A'
        print(f"{rank:4} | {row['comment_id']:7} | {row['variance_score']:8.1f} | {text_preview}")
    
    result.to_csv('selected_comments.csv', index=False)
    print(f'\n💾 Results saved to selected_comments.csv')
    print(f"📊 Included all {len(result.columns)} columns from processed_dataset.csv")
    print(f"🎯 These represent the top {len(result)} most demographically diverse comments")
    print(f"📈 Variance shows how diverse the annotator demographics were for each comment")

if __name__ == "__main__":
    main()