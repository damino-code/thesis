# Dynamic Prompts Implementation Guide

This document explains how the dynamic prompts system is implemented in the codebase.

---

## Architecture Diagram

```
run_all_attributes.py
    ↓
[User selects mode: 1=Vanilla, 2=Feature]
    ↓
shared_runner.py.run_attribute_analysis(use_dynamic=True/False, comment_id)
    ↓
single_attribute_analyzer.py.SingleAttributeAnalyzer(use_dynamic=True/False)
    ├─ If use_dynamic=True:
    │   └─ Load annotator_features_loader.py
    │       └─ Read selected_comments.csv
    │
    └─ analyze_attribute(comment_text, attribute, comment_id)
        ├─ If use_dynamic=True and comment_id provided:
        │   ├─ Get features: features_loader.get_features(comment_id)
        │   ├─ Build prompt: _build_dynamic_persona(comment_id)
        │   └─ Pass system_message to LLM
        │
        └─ If use_dynamic=False:
            └─ Use standard system message
```

---

## Code Walkthrough

### 1. User Mode Selection (run_all_attributes.py)

```python
import sys
from shared_runner import run_attribute_analysis

# User menu
print("\n=== Analysis Mode Selection ===")
print("1. Vanilla (Standard static prompts)")
print("2. Feature (Dynamic annotator-specific prompts)")
mode_input = input("Select mode (1/2): ").strip()

use_dynamic = (mode_input == "2")

# Pass to runner
attributes = ["hate_speech", "sentiment", "respect", "offensiveness"]
for attr in attributes:
    run_attribute_analysis(
        attribute_name=attr,
        sample_size='all',
        use_dynamic=use_dynamic  # ← Flows through entire pipeline
    )
```

---

### 2. Feature Loader (annotator_features_loader.py)

```python
import pandas as pd
import os

class AnnotatorFeaturesLoader:
    """Loads and caches annotator demographic features from CSV"""
    
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = "/storage/home/amine/thesis/Dataset/selected_comments.csv"
        
        self.csv_path = csv_path
        self.features_cache = {}  # ← In-memory cache
        self.features_df = None
    
    def load_features(self):
        """Load CSV and prepare for lookups"""
        if self.features_df is None:
            self.features_df = pd.read_csv(self.csv_path)
            self.features_df = self.features_df.set_index('comment_id')
        return self.features_df
    
    def get_features(self, comment_id):
        """
        Get annotator features for a specific comment_id
        Returns dict with keys: gender, age, race, religion, ideology
        """
        # Check cache first
        if comment_id in self.features_cache:
            return self.features_cache[comment_id]
        
        # Load if not loaded
        if self.features_df is None:
            self.load_features()
        
        # Lookup
        try:
            row = self.features_df.loc[comment_id]
            features = {
                'gender': str(row.get('annotator_gender', 'Not specified')),
                'age': float(row.get('annotator_age', 0)),
                'race': str(row.get('annotator_race', 'Not specified')),
                'religion': str(row.get('annotator_religion', 'Not specified')),
                'ideology': str(row.get('annotator_ideology', 'Not specified'))
            }
            # Handle NaN
            for key in features:
                if pd.isna(features[key]):
                    features[key] = 'Not specified'
        except KeyError:
            features = {k: 'Not specified' for k in ['gender', 'age', 'race', 'religion', 'ideology']}
        
        # Cache
        self.features_cache[comment_id] = features
        return features
```

---

### 3. Analyzer with Dynamic Prompts (single_attribute_analyzer.py - Key Sections)

```python
from annotator_features_loader import AnnotatorFeaturesLoader

class SingleAttributeAnalyzer:
    
    def __init__(self, llm_model, use_dynamic=False):
        self.llm_model = llm_model
        self.use_dynamic = use_dynamic
        
        # Only load features if dynamic mode
        if use_dynamic:
            self.features_loader = AnnotatorFeaturesLoader()
            self.features_loader.load_features()
        else:
            self.features_loader = None
    
    def _build_dynamic_persona(self, comment_id):
        """
        Build a system prompt that includes annotator demographics
        """
        if not self.use_dynamic or not self.features_loader:
            return None
        
        features = self.features_loader.get_features(comment_id)
        
        system_message = f"""You are a content annotator with the following characteristics:
- Gender: {features['gender']}
- Age: {features['age']}
- Race: {features['race']}
- Religion: {features['religion']}
- Ideology: {features['ideology']}

Evaluate this comment from your perspective as this annotator."""
        
        return system_message
    
    def analyze_attribute(self, text, attribute, comment_id=None):
        """
        Core analysis method
        """
        # Build system message
        if self.use_dynamic and comment_id is not None:
            system_message = self._build_dynamic_persona(comment_id)
        else:
            system_message = self._build_standard_system_message()
        
        # Build user message (standard scoring criteria + comment)
        user_message = self._build_user_message(attribute, text)
        
        # Call LLM with the appropriate system message
        response = self.llm_model.generate(
            system=system_message,
            user=user_message
        )
        
        return response
    
    def _build_standard_system_message(self):
        return "You are an expert content annotator."
    
    def _build_user_message(self, attribute, text):
        # Prompts are loaded from JSON files in prompts/ folder
        # These contain the actual scoring criteria:
        criteria = {
            'hatespeech': 'HATESPEECH (0-2): Question: Does this comment contain hate speech? 0=yes , 1=No, 2=Unclear/Neutral',
            'sentiment': 'SENTIMENT (0-4): Question: How would you describe the sentiment? 0=Strongly negative, 1=Somewhat negative, 2=Neutral, 3=Somewhat positive, 4=Strongly positive',
            'respect': 'RESPECT (0-4): Question: Is this comment respectful towards the group? 0=Strongly disrespectful, 1=Disrespectful/Rude, 2=Neutral, 3=Respectful/Polite, 4=Strongly respectful',
            'dehumanize': 'DEHUMANIZE (0-4): Question: Does the comment portray the group as less than human? 0=Strongly no, 1=No, 2=Unclear/Neutral, 3=Yes, 4=Strongly yes',
            'attack_defend': 'ATTACK_DEFEND (0-4): Question: Is the comment attacking or defending the group? 0=Strongly defending, 1=Defending, 2=Neutral/Mixed, 3=Attacking, 4=Strongly attacking',
            'violence': 'VIOLENCE (0-4): Question: Does the comment call for violence against the group? 0=Strongly no, 1=No, 2=Unclear/Neutral, 3=Yes, 4=Strongly yes',
            'genocide': 'GENOCIDE (0-4): Question: Does the comment call for the deliberate killing of a large group? 0=Strongly no, 1=No, 2=Unclear/Neutral, 3=Yes, 4=Strongly yes'
        }
        
        attr_name = attribute.lower()
        scoring = criteria.get(attr_name, 'ATTRIBUTE (0-4): Evaluate this attribute')
        
        return f"""SCORING CRITERIA (Answer based on the definitions below):
- {scoring}

Analyze this comment for {attribute.upper()}:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER."""
```

---

### 4. Shared Runner (shared_runner.py - Key Sections)

```python
def run_attribute_analysis(attribute_name, sample_size='all', use_dynamic=False):
    """
    Execute analysis pipeline for a single attribute
    """
    # Load data
    df = load_data()
    
    # Create analyzer (with use_dynamic flag)
    analyzer = SingleAttributeAnalyzer(
        llm_model=model,
        use_dynamic=use_dynamic  # ← Pass flag
    )
    
    # Conditional folder path based on mode
    if use_dynamic:
        result_folder = f"results/persona_results/{attribute_name}"
    else:
        result_folder = f"results/single_attribute_analyser/{attribute_name}"
    
    os.makedirs(result_folder, exist_ok=True)
    
    # Process each comment
    results = []
    for idx, row in df.iterrows():
        text = row['comment_text']
        
        # Conditional comment_id extraction (only if dynamic)
        comment_id = row.get('comment_id', idx) if use_dynamic else None
        
        # Analyze with optional comment_id
        prediction = analyzer.analyze_attribute(
            text=text,
            attribute=attribute_name,
            comment_id=comment_id  # ← Only used if use_dynamic=True
        )
        
        results.append({
            'comment_id': row.get('comment_id', idx),
            'prediction': prediction,
            'true_label': row[f'human_{attribute_name}']
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{result_folder}/results_{attribute_name}_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"Saved {attribute_name} results to {output_path}")
```

---

### 5. Result Merging (merge_results.py - Key Sections)

```python
def merge_results(use_dynamic=False):
    """
    Merge individual attribute results into one file
    """
    # Conditional base directory
    if use_dynamic:
        search_base_dir = "results/persona_results"
        output_name = "merged_persona_results"
    else:
        search_base_dir = "results/single_attribute_analyser"
        output_name = "merged_standard_results"
    
    # Find all result files matching the mode
    all_files = glob.glob(f"{search_base_dir}/**/results_*.csv", recursive=True)
    
    # Concat all attribute results
    merged_df = pd.concat([pd.read_csv(f) for f in all_files], axis=1)
    
    # Save with appropriate naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/{output_name}_{timestamp}.csv"
    merged_df.to_csv(output_path, index=False)
    
    print(f"Merged results saved to {output_path}")
```

---

### 6. Evaluation Mode Selection (evaluation.py - Key Sections)

```python
def evaluation_main():
    """
    User selects which mode's results to evaluate
    """
    print("\n=== Evaluation Mode Selection ===")
    print("1. Standard (Vanilla static prompts)")
    print("2. Persona (Dynamic annotator-specific prompts)")
    mode_input = input("Select mode to evaluate (1/2): ").strip()
    
    if mode_input == "1":
        # Load vanilla results
        merged_file = glob.glob("results/merged_standard_results_*.csv")[-1]
        results_df = pd.read_csv(merged_file)
        print(f"Evaluating Standard results from {merged_file}")
    else:
        # Load dynamic results
        merged_file = glob.glob("results/merged_persona_results_*.csv")[-1]
        results_df = pd.read_csv(merged_file)
        print(f"Evaluating Persona results from {merged_file}")
    
    # Run evaluation
    metrics = calculate_metrics(results_df)
    create_visualizations(metrics)
```

---

## Data Flow Example

**Scenario:** Analyzing comment_id 39073 for hate_speech attribute

### Step 1: User Selects Mode
```
User input: 2 (Feature mode)
use_dynamic = True
```

### Step 2: Analyzer Initializes with Features Loader
```python
analyzer = SingleAttributeAnalyzer(llm_model, use_dynamic=True)
# analyzer.features_loader now holds selected_comments.csv data
```

### Step 3: Processing Comment
```python
prediction = analyzer.analyze_attribute(
    text="Afghanistan will rise...",
    attribute="hate_speech",
    comment_id=39073  # ← Provided because use_dynamic=True
)
```

### Step 4: Dynamic Persona Built
```python
features = features_loader.get_features(39073)
# Returns: {'gender': 'Women', 'age': 81.0, 'race': 'White', ...}

system_message = f"""You are a content moderator with the following characteristics:
- Gender: Women
- Age: 81.0
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator."""
```

### Step 5: LLM Called
```
LLM receives:
- system: [dynamic persona message with this woman's features]
- user: "Is this hate speech? Comment: Afghanistan will rise..."
Returns: 2 (from Conservative Christian woman's perspective)
```

### Step 6: Results Saved
```
File path: results/persona_results/hate_speech/results_hate_speech_20250115_120530.csv
Contains rows with:
- comment_id: 39073
- prediction: 2
- true_label: 2 (matches this annotator's human label)
```

---

## Performance Considerations

1. **CSV Loading:** Loaded once at analyzer init, reused for all comments
2. **Feature Caching:** Dictionary cache prevents repeated CSV lookups
3. **Comment_id Optional:** Code gracefully handles missing comment_id (falls back to vanilla)
4. **Memory Efficient:** Features cache grows with unique comment_ids, typically < 50KB for dataset

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| comment_id not in CSV | Returns NaN → displays as "Not specified" |
| Missing column in CSV | Returns NaN → displays as "Not specified" |
| use_dynamic=False | Skips feature loading, uses standard prompt |
| Comment_id=None in dynamic mode | Uses standard prompt as fallback |

---

## Testing the Implementation

```python
# Test 1: Feature loading
from annotator_features_loader import AnnotatorFeaturesLoader
loader = AnnotatorFeaturesLoader()
features = loader.get_features(39073)
print(features)  # Should show: {'gender': 'Women', 'age': 81.0, ...}

# Test 2: Dynamic persona building
from single_attribute_analyzer import SingleAttributeAnalyzer
analyzer = SingleAttributeAnalyzer(llm_model, use_dynamic=True)
prompt = analyzer._build_dynamic_persona(39073)
print(prompt)  # Should show persona with this comment's annotator features

# Test 3: Vanilla vs Dynamic
prompt_vanilla = analyzer._build_standard_system_message()
prompt_dynamic = analyzer._build_dynamic_persona(39073)
assert prompt_vanilla != prompt_dynamic  # Should be different
```

---

## Verification Commands

```bash
# Verify feature loader works
python -c "from annotator_features_loader import AnnotatorFeaturesLoader; l=AnnotatorFeaturesLoader(); l.load_features(); print(l.get_features(39073))"

# Run vanilla mode analysis
python src/run_all_attributes.py
# Select: 1

# Run dynamic mode analysis
python src/run_all_attributes.py
# Select: 2

# Check both result folders
ls -la results/single_attribute_analyser/
ls -la results/persona_results/

# Compare merged results
diff <(head results/merged_standard_results_*.csv) <(head results/merged_persona_results_*.csv)
```

---

## Key Implementation Principles

1. **Single Source of Truth:** Features come only from selected_comments.csv
2. **Graceful Degradation:** Missing data → "Not specified", not errors
3. **Mode Isolation:** vanilla and dynamic results never mix
4. **Conditional Complexity:** use_dynamic flag gates all feature logic
5. **Transparent Caching:** Features cached transparently, user doesn't see this

