import pandas as pd
import os

class AnnotatorFeaturesLoader:
    """Load and cache annotator features from selected_comments.csv"""
    
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = "/storage/home/amine/thesis/Dataset/selected_comments.csv"
        self.csv_path = csv_path
        self.features_df = None
        self.features_cache = {}
    
    def load_features(self):
        """Load CSV and cache feature columns"""
        if not os.path.exists(self.csv_path):
            print(f"⚠️  Warning: CSV file not found at {self.csv_path}")
            self.features_df = pd.DataFrame()
            return
        
        self.features_df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.features_df)} annotator records")
    
    def get_features(self, comment_id):
        """Get annotator features for a specific comment_id"""
        if self.features_df is None or self.features_df.empty:
            return None
        
        # Check cache first
        if comment_id in self.features_cache:
            return self.features_cache[comment_id]
        
        # Query features
        record = self.features_df[self.features_df['comment_id'] == comment_id]
        
        if len(record) == 0:
            return None  # Comment not found
        
        row = record.iloc[0]
        
        # Extract the 5 features we need
        features = {
            'gender': row.get('annotator_gender', 'Unknown'),
            'age': row.get('annotator_age', 'Unknown'),
            'race': row.get('annotator_race', 'Unknown'),
            'religion': row.get('annotator_religion', 'Unknown'),
            'ideology': row.get('annotator_ideology', 'Unknown')
        }
        
        # Handle NaN values
        features = {k: (str(v) if pd.notna(v) else 'Not specified') 
                   for k, v in features.items()}
        
        # Determine age category: old if > 50, young if <= 50
        try:
            age_value = float(features['age']) if features['age'] != 'Not specified' else None
            if age_value is not None:
                features['age_category'] = 'old' if age_value > 50 else 'young'
            else:
                features['age_category'] = 'Not specified'
        except (ValueError, TypeError):
            features['age_category'] = 'Not specified'
        
        # Cache it
        self.features_cache[comment_id] = features
        return features
