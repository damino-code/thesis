import pandas as pd
import os


class AnnotatorFeaturesLoader:
    """Load and cache annotator features from processed_dataset.csv"""

    def __init__(self, csv_path=None):
        if csv_path is None:
            # src/ -> model/ -> HateSpeechPromptScoreFullAnnotation/ -> thesis/
            src_dir = os.path.dirname(os.path.abspath(__file__))
            thesis_dir = os.path.dirname(os.path.dirname(os.path.dirname(src_dir)))
            csv_path = os.path.join(thesis_dir, "Dataset", "processed_dataset.csv")
        self.csv_path = csv_path
        self.features_df = None
        self.features_cache = {}

    def load_features(self):
        if not os.path.exists(self.csv_path):
            print(f"Warning: CSV file not found at {self.csv_path}")
            self.features_df = pd.DataFrame()
            return
        self.features_df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.features_df)} annotator records from {self.csv_path}")

    def get_features(self, comment_id, annotator_id=None):
        if self.features_df is None or self.features_df.empty:
            return None

        cache_key = (comment_id, annotator_id)
        if cache_key in self.features_cache:
            return self.features_cache[cache_key]

        mask = self.features_df['comment_id'] == comment_id
        if annotator_id is not None:
            mask = mask & (self.features_df['annotator_id'] == annotator_id)

        record = self.features_df[mask]

        if len(record) == 0:
            print(f"No record found for comment_id={comment_id}, annotator_id={annotator_id}")
            return None

        row = record.iloc[0]

        features = {
            'gender':   row.get('annotator_gender',   'Not specified'),
            'age':      row.get('annotator_age',      'Not specified'),
            'race':     row.get('annotator_race',     'Not specified'),
            'religion': row.get('annotator_religion', 'Not specified'),
            'ideology': row.get('annotator_ideology', 'Not specified'),
        }

        features = {k: (str(v) if pd.notna(v) else 'Not specified')
                    for k, v in features.items()}

        try:
            age_value = float(features['age']) if features['age'] != 'Not specified' else None
            features['age_category'] = ('old' if age_value is not None and age_value > 50
                                        else 'young' if age_value is not None
                                        else 'Not specified')
        except (ValueError, TypeError):
            features['age_category'] = 'Not specified'

        self.features_cache[cache_key] = features
        return features
