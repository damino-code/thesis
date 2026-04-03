# Dynamic Prompts - Code Snippet Reference

## 🔍 Where to Find the Dynamic Prompt Code

---

## 1️⃣ Feature Loading Code
**File:** `annotator_features_loader.py` (in `/src/` folder)

### How it works:
```python
# LINE 28-46: Get features for a specific comment_id
def get_features(self, comment_id):
    """Get annotator features for a specific comment_id"""
    if self.features_df is None or self.features_df.empty:
        return None
    
    # Check cache first
    if comment_id in self.features_cache:
        return self.features_cache[comment_id]
    
    # Query features from CSV
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
    
    # IMPORTANT: Determine age category
    # ✓ old if > 50, ✓ young if ≤ 50
    try:
        age_value = float(features['age']) if features['age'] != 'Not specified' else None
        if age_value is not None:
            features['age_category'] = 'old' if age_value > 50 else 'young'
        else:
            features['age_category'] = 'Not specified'
    except (ValueError, TypeError):
        features['age_category'] = 'Not specified'
    
    # Cache it for performance
    self.features_cache[comment_id] = features
    return features
```

**Result:** Returns dict like:
```python
{
  'gender': 'Women',
  'age': '81.0',
  'age_category': 'old',  # ← Automatically calculated!
  'race': 'White',
  'religion': 'Christian',
  'ideology': 'Conservative'
}
```

---

## 2️⃣ Dynamic Persona Building
**File:** `single_attribute_analyzer.py` (in `/src/` folder)

### Lines 46-62: Build system message with features

```python
def _build_dynamic_persona(self, comment_id):
    """Build system prompt with annotator features based on comment_id"""
    if comment_id is None or self.features_loader is None:
        return "You are an expert content annotator."
    
    # Step 1: Get features for this comment_id
    features = self.features_loader.get_features(comment_id)
    
    if features is None:
        return "You are an expert content annotator."
    
    # Step 2: Build contextual prompt with annotator characteristics
    prompt = f"""You are a content annotator with the following characteristics:
- Gender: {features['gender']}
- Age: {features['age']}
- Race: {features['race']}
- Religion: {features['religion']}
- Ideology: {features['ideology']}

Evaluate this comment from your perspective as this annotator."""
    
    return prompt
```

**This produces:**
```
You are a content annotator with the following characteristics:
- Gender: Women
- Age: 81.0
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator.
```

---

## 3️⃣ Per-Feature Analysis with Dynamic Prompts
**File:** `single_attribute_analyzer.py` (Lines 64-95)

### How each attribute gets evaluated with dynamic prompts:

```python
def analyze_attribute(self, text, attribute, comment_id=None):
    """Analyze a comment for a single attribute"""
    if attribute not in self.prompts:
        raise ValueError(f"Unknown attribute: {attribute}")
    
    # Step 1: Decide which system prompt to use
    if self.use_dynamic:
        # ✓ DYNAMIC: Uses features from CSV
        persona_prefix = self._build_dynamic_persona(comment_id)
    else:
        # ✗ VANILLA: Generic prompt
        persona_prefix = "You are an expert content annotator."
    
    # Step 2: Build the full prompt with system + user message
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{persona_prefix}<|eot_id|><|start_header_id|>user<|end_header_id|>
{self.prompts[attribute].format(text=text[:500])}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    # Step 3: Send to LLM
    try:
        response = self.model(
            full_prompt,
            max_tokens=10,
            temperature=0.1,
            stop=["<|eot_id|>"],
            logprobs=1
        )
        # Extract prediction...
        return {attribute: val, 'confidence': confidence}
```

**For each attribute (HATESPEECH, SENTIMENT, RESPECT, etc.):**
1. Load the scoring criteria from `prompts/{attribute}.json`
2. Insert the dynamic persona system message
3. Add the scoring criteria user message
4. Send to LLM → Get personalized prediction

---

## 🔄 Data Flow for Dynamic Prompts

```
COMMENT with comment_id=39073
    ↓
single_attribute_analyzer.analyze_attribute(text, "sentiment", comment_id=39073)
    ↓
[IF use_dynamic=True]:
  └─→ _build_dynamic_persona(39073)
      └─→ features_loader.get_features(39073)
          ├─ Query CSV row where comment_id=39073
          ├─ Extract: {gender, age, race, religion, ideology}
          ├─ Calculate: age_category = 'old' (age 81 > 50)
          └─ Return: full features dict
      └─→ Build system prompt with these features
          └─ "You are a content annotator with the following characteristics:
             - Gender: Women
             - Age: 81.0
             - Race: White
             - Religion: Christian
             - Ideology: Conservative
             
             Evaluate this comment from your perspective as this annotator."
    ↓
Combined prompt sent to LLM:
    System: [Dynamic persona from above]
    User: [SENTIMENT criteria + comment text]
    ↓
LLM Response: "0" (Strongly negative - from this Conservative woman's perspective)
```

---

## 📊 Feature-Specific Examples

### Example: SENTIMENT attribute
**Code location:** `prompts/sentiment.json`

```json
{
    "prompt": [
        "SCORING CRITERIA (Answer based on the definitions below):",
        "- SENTIMENT (0-4): Question: How would you describe the sentiment? 0=Strongly negative, 1=Somewhat negative, 2=Neutral, 3=Somewhat positive, 4=Strongly positive",
        "",
        "Analyze this comment for SENTIMENT:",
        "Comment: \"{text}\"",
        "",
        "RESPOND WITH ONLY THE NUMBER (0-4)."
    ]
}
```

**Full prompt sent to LLM for SENTIMENT with dynamic persona:**
```
System: You are a content annotator with the following characteristics:
- Gender: Women
- Age: 81.0
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator.

User: SCORING CRITERIA (Answer based on the definitions below):
- SENTIMENT (0-4): Question: How would you describe the sentiment? 0=Strongly negative, 1=Somewhat negative, 2=Neutral, 3=Somewhat positive, 4=Strongly positive

Analyze this comment for SENTIMENT:
Comment: "Afghanistan will rise. Kashmir will rise. Palestine will rise..."

RESPOND WITH ONLY THE NUMBER (0-4).
```

**Output:** `0` (Strongly negative from her perspective)

---

### Example: HATESPEECH attribute
**Code location:** `prompts/hatespeech.json`

```json
{
    "prompt": [
        "SCORING CRITERIA (Answer based on the definitions below):",
        "- HATESPEECH (0-2): Question: Does this comment contain hate speech? 0=yes , 1=No, 2=Unclear/Neutral",
        "",
        "Analyze this comment for HATE SPEECH:",
        "Comment: \"{text}\"",
        "",
        "RESPOND WITH ONLY THE NUMBER (0-2)."
    ]
}
```

**Full prompt sent to LLM for HATESPEECH with dynamic persona:**
```
System: You are a content annotator with the following characteristics:
- Gender: Women
- Age: 25.0
- Race: Black
- Religion: Christian
- Ideology: Liberal

Evaluate this comment from your perspective as this annotator.

User: SCORING CRITERIA (Answer based on the definitions below):
- HATESPEECH (0-2): Question: Does this comment contain hate speech? 0=yes , 1=No, 2=Unclear/Neutral

Analyze this comment for HATE SPEECH:
Comment: "Black people are taking over our country"

RESPOND WITH ONLY THE NUMBER (0-2).
```

**Output:** `0` (yes, definitely hate speech - from young Black woman's perspective)

---

### Example: RESPECT attribute  
**Code location:** `prompts/respect.json`

```json
{
    "prompt": [
        "SCORING CRITERIA (Answer based on the definitions below):",
        "- RESPECT (0-4): Question: Is this comment respectful towards the group? 0=Strongly disrespectful, 1=Disrespectful/Rude, 2=Neutral, 3=Respectful/Polite, 4=Strongly respectful",
        "",
        "Analyze this comment for RESPECT:",
        "Comment: \"{text}\"",
        "",
        "RESPOND WITH ONLY THE NUMBER (0-4)."
    ]
}
```

**Full prompt with Conservative woman's persona:**
```
System: ... [Dynamic persona with Conservative elderly woman's features]
User: RESPECT criteria + comment
Output: Likely 0 or 1 (disrespectful) if discussing her demographic
```

---

## 🎯 All Supported Attributes with their Criteria

| Attribute | File | Scale | Criteria |
|-----------|------|-------|----------|
| **HATESPEECH** | `prompts/hatespeech.json` | 0-2 | Does comment contain hate speech? 0=yes, 1=No, 2=Unclear |
| **SENTIMENT** | `prompts/sentiment.json` | 0-4 | How would you describe sentiment? 0=Strongly negative → 4=Strongly positive |
| **RESPECT** | `prompts/respect.json` | 0-4 | Is comment respectful? 0=Strongly disrespectful → 4=Strongly respectful |
| **DEHUMANIZE** | `prompts/dehumanize.json` | 0-4 | Portray as less than human? 0=Strongly no → 4=Strongly yes |
| **ATTACK_DEFEND** | `prompts/attack_defend.json` | 0-4 | Attacking or defending? 0=Strongly defending → 4=Strongly attacking |
| **VIOLENCE** | `prompts/violence.json` | 0-4 | Call for violence? 0=Strongly no → 4=Strongly yes |
| **GENOCIDE** | `prompts/genocide.json` | 0-4 | Call for genocide? 0=Strongly no → 4=Strongly yes |

---

## 🔧 How to Trace Dynamic Prompts in Code

### Start here → Follow the bouncing ball:

1. **User runs analysis:**
   ```bash
   python src/run_all_attributes.py
   # Selects: 2 (Feature mode)
   ```

2. **Mode selection flows here:**
   📄 `src/run_all_attributes.py` (Lines 1-30)
   - Sets `use_dynamic = True`
   - Calls `run_attribute_analysis(..., use_dynamic=True)`

3. **Orchestration layer:**
   📄 `src/shared_runner.py` (Lines 1-50)
   - Receives `use_dynamic=True`
   - Passes to SingleAttributeAnalyzer

4. **Feature loading:**
   📄 `src/single_attribute_analyzer.py` Line 17-21
   - Initializes: `self.features_loader = AnnotatorFeaturesLoader()`
   - Loads CSV: `self.features_loader.load_features()`

5. **Per-comment analysis:**
   📄 `src/single_attribute_analyzer.py` Line 64-95
   - For each comment, calls: `analyze_attribute(text, attribute, comment_id)`
   - Calls: `_build_dynamic_persona(comment_id)`

6. **Feature retrieval:**
   📄 `src/annotator_features_loader.py` Line 28-46
   - Looks up comment_id in CSV
   - Returns features + age_category

7. **System message built:**
   📄 `src/single_attribute_analyzer.py` Line 46-62
   - Injects features into template
   - Returns system prompt

8. **LLM receives:**
   - System: Dynamic persona message
   - User: Scoring criteria from `prompts/{attribute}.json`
   - Comment: Actual comment text

---

## 💡 Key Code Locations Summary

| What | Where | Lines |
|------|-------|-------|
| **Feature Loading** | `annotator_features_loader.py` | 28-46 |
| **Age Category Calculation** | `annotator_features_loader.py` | 39-43 |
| **Dynamic Persona Building** | `single_attribute_analyzer.py` | 46-62 |
| **Per-Attribute Analysis** | `single_attribute_analyzer.py` | 64-95 |
| **Vanilla vs Dynamic Logic** | `single_attribute_analyzer.py` | 70-77 |
| **Mode Selection** | `run_all_attributes.py` | 1-30 |
| **Scoring Criteria** | `prompts/*.json` | (all files) |

