# Attribute-Specific Dynamic Prompts Architecture

## Overview
Each attribute now has its **own dynamic prompt template**, just like it has its own standard prompt. This ensures that each attribute's scoring criteria is personalized to the annotator's demographics.

---

## File Structure

```
src/
├── prompts/                           # Standard prompts (unchanged)
│   ├── hatespeech.json               # Score: 0-2
│   ├── sentiment.json                 # Score: 0-4
│   ├── respect.json                   # Score: 0-4
│   ├── dehumanize.json               # Score: 0-4
│   ├── attack_defend.json            # Score: 0-4
│   ├── violence.json                 # Score: 0-4
│   ├── genocide.json                 # Score: 0-4
│   │
│   └── dynamic/                       # NEW: Dynamic prompts (one per attribute!)
│       ├── hatespeech.json           # Dynamic version with persona + criteria
│       ├── sentiment.json             # Dynamic version with persona + criteria
│       ├── respect.json               # Dynamic version with persona + criteria
│       ├── dehumanize.json           # Dynamic version with persona + criteria
│       ├── attack_defend.json        # Dynamic version with persona + criteria
│       ├── violence.json              # Dynamic version with persona + criteria
│       └── genocide.json              # Dynamic version with persona + criteria
│
├── single_attribute_analyzer.py      # Updated with new methods
├── annotator_features_loader.py      # (unchanged)
└── shared_runner.py                   # (unchanged)
```

---

## How It Works

### Vanilla (Standard) Mode
```
Comment ID: 39073
    ↓
analyze_attribute(text, "sentiment", comment_id=None)
    ↓
Loads: prompts/sentiment.json
    ↓
System: "You are an expert content annotator."
User: "SENTIMENT (0-4): ... Analyze: {text} ..."
    ↓
LLM Response: Generic perspective
```

### Dynamic (Feature-based) Mode
```
Comment ID: 39073
    ↓
analyze_attribute(text, "sentiment", comment_id=39073)
    ↓
_build_dynamic_prompt("sentiment", 39073, text)
    ├─ Get features from CSV (gender, age, age_category, race, religion, ideology)
    ├─ Load: prompts/dynamic/sentiment.json
    ├─ Format template with features
    └─ Return complete personalized prompt
    ↓
Full User Prompt (includes persona + sentiment-specific criteria):
"You are a content annotator with the following characteristics:
- Gender: Women
- Age: 81.0 (old)
- Race: White
- Religion: Christian
- Ideology: Conservative

SENTIMENT (0-4): How would you describe the sentiment?...
Comment: {text}
RESPOND WITH ONLY THE NUMBER (0-4)."
    ↓
LLM Response: From Conservative old woman's perspective
```

---

## Code Implementation

### New Methods in `single_attribute_analyzer.py`

#### 1. Load Dynamic Prompts (Lines 30-50)
```python
def _load_dynamic_prompts(self):
    """Load attribute-specific dynamic prompt templates"""
    dynamic_prompts = {}
    dynamic_prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts', 'dynamic')
    
    for filename in os.listdir(dynamic_prompt_dir):
        if filename.endswith(".json"):
            attribute = os.path.splitext(filename)[0]
            with open(os.path.join(dynamic_prompt_dir, filename), 'r') as f:
                data = json.load(f)
                # Each file has a "dynamic_prompt" key
                if isinstance(data['dynamic_prompt'], list):
                    dynamic_prompts[attribute] = '\n'.join(data['dynamic_prompt'])
                else:
                    dynamic_prompts[attribute] = data['dynamic_prompt']
    
    return dynamic_prompts
```

#### 2. Build Dynamic Prompt (Lines 102-131)
```python
def _build_dynamic_prompt(self, attribute, comment_id, text):
    """Build complete dynamic prompt with features injected for a specific attribute"""
    if attribute not in self.dynamic_prompts:
        return None
    
    # Get features from CSV
    features = self.features_loader.get_features(comment_id)
    if features is None:
        return None
    
    # Format dynamic prompt template with features and text
    dynamic_template = self.dynamic_prompts[attribute]
    
    try:
        formatted_prompt = dynamic_template.format(
            gender=features['gender'],
            age=features['age'],
            age_category=features.get('age_category', 'Not specified'),
            race=features['race'],
            religion=features['religion'],
            ideology=features['ideology'],
            text=text[:500]
        )
        return formatted_prompt
    except KeyError as e:
        return None
```

#### 3. Updated analyze_attribute (Lines 133-180)
```python
def analyze_attribute(self, text, attribute, comment_id=None):
    """Analyze a comment for a single attribute"""
    if attribute not in self.prompts:
        raise ValueError(f"Unknown attribute: {attribute}")
    
    # Build prompt - either dynamic or vanilla
    if self.use_dynamic and comment_id is not None:
        # Use ATTRIBUTE-SPECIFIC dynamic prompt
        user_message = self._build_dynamic_prompt(attribute, comment_id, text)
        if user_message is None:
            # Fallback to vanilla
            user_message = self.prompts[attribute].format(text=text[:500])
            system_message = "You are an expert content annotator."
        else:
            system_message = ""  # All content in user message
    else:
        # Use standard vanilla prompt
        system_message = "You are an expert content annotator."
        user_message = self.prompts[attribute].format(text=text[:500])
    
    # Build full prompt with system + user messages
    # ... send to LLM
```

---

## Dynamic Prompt Template Format

### Example: `prompts/dynamic/sentiment.json`

```json
{
    "dynamic_prompt": [
        "You are a content annotator with the following characteristics:",
        "- Gender: {gender}",
        "- Age: {age} ({age_category})",
        "- Race: {race}",
        "- Religion: {religion}",
        "- Ideology: {ideology}",
        "",
        "Evaluate this comment from your perspective as this annotator.",
        "",
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

**Placeholders that get formatted:**
- `{gender}` → "Women", "Men", etc.
- `{age}` → "81.0", "40.0", etc.
- `{age_category}` → "old" (>50) or "young" (≤50)
- `{race}` → "White", "Black", "Asian", etc.
- `{religion}` → "Christian", "Nothing", "Atheist", etc.
- `{ideology}` → "Conservative", "Liberal", etc.
- `{text}` → The actual comment (first 500 chars)

---

## Attribute-Specific Examples

### HATESPEECH with Dynamic Prompt
**Attribute:** hatespeech
**Scale:** 0-2

**Template File:** `prompts/dynamic/hatespeech.json`

**Generated Prompt (for 81-year-old Conservative woman):**
```
You are a content annotator with the following characteristics:
- Gender: Women
- Age: 81.0 (old)
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator.

SCORING CRITERIA (Answer based on the definitions below):
- HATESPEECH (0-2): Question: Does this comment contain hate speech? 0=yes , 1=No, 2=Unclear/Neutral

Analyze this comment for HATE SPEECH:
Comment: "Black people are taking over our country"

RESPOND WITH ONLY THE NUMBER (0-2).
```

---

### RESPECT with Dynamic Prompt
**Attribute:** respect
**Scale:** 0-4

**Template File:** `prompts/dynamic/respect.json`

**Generated Prompt (for 25-year-old Liberal young Black woman):**
```
You are a content annotator with the following characteristics:
- Gender: Women
- Age: 25.0 (young)
- Race: Black
- Religion: Christian
- Ideology: Liberal

Evaluate this comment from your perspective as this annotator.

SCORING CRITERIA (Answer based on the definitions below):
- RESPECT (0-4): Question: Is this comment respectful towards the group? 0=Strongly disrespectful, 1=Disrespectful/Rude, 2=Neutral, 3=Respectful/Polite, 4=Strongly respectful

Analyze this comment for RESPECT:
Comment: "Black people are lazy"

RESPOND WITH ONLY THE NUMBER (0-4).
```

---

## All 7 Attributes with Dynamic Prompts

| Attribute | File | Scale | Key Difference |
|-----------|------|-------|-----------------|
| **HATESPEECH** | `dynamic/hatespeech.json` | 0-2 | Does comment contain hate speech? |
| **SENTIMENT** | `dynamic/sentiment.json` | 0-4 | How would YOU describe sentiment? |
| **RESPECT** | `dynamic/respect.json` | 0-4 | Is comment respectful to YOUR group? |
| **DEHUMANIZE** | `dynamic/dehumanize.json` | 0-4 | Does comment portray YOUR group as less human? |
| **ATTACK_DEFEND** | `dynamic/attack_defend.json` | 0-4 | Does comment attack or defend YOUR group? |
| **VIOLENCE** | `dynamic/violence.json` | 0-4 | Does comment call for violence against YOUR group? |
| **GENOCIDE** | `dynamic/genocide.json` | 0-4 | Does comment call for genocide of YOUR group? |

---

## Key Difference from Before

### ❌ OLD Approach (Generic Dynamic Persona)
```
_build_dynamic_persona(comment_id)  # Generic for ALL attributes
    ↓
Always returns same system message with features
    ↓
Then appends standard prompt from prompts/{attribute}.json
    ↓
Result: Not truly personalized per attribute
```

### ✅ NEW Approach (Attribute-Specific Dynamic Prompts)
```
_build_dynamic_prompt(attribute, comment_id, text)  # Specific per attribute
    ↓
Loads dynamic prompt template from prompts/dynamic/{attribute}.json
    ↓
Formats template with features (gender, age_category, race, etc.)
    ↓
Result: Full personalized prompt with persona + attribute-specific criteria
    ↓
Each attribute has its own dynamic "voice"
```

---

## How to Verify It Works

### Check Available Dynamic Prompts
```bash
ls -la src/prompts/dynamic/
# Should show 7 files: hatespeech.json, sentiment.json, respect.json, etc.
```

### Trace Code Execution
1. **Mode Selection:** `src/run_all_attributes.py` (user selects 2 for dynamic)
2. **Initialization:** `src/single_attribute_analyzer.py:__init__` → calls `_load_dynamic_prompts()`
3. **Per-Attribute:** `src/single_attribute_analyzer.py:analyze_attribute()` → calls `_build_dynamic_prompt()`
4. **Template Loaded:** From `src/prompts/dynamic/{attribute}.json`
5. **Features Injected:** From annotator_features_loader
6. **LLM Receives:** Complete personalized prompt

---

## Backward Compatibility

✅ **Vanilla mode still works:** Uses standard `prompts/{attribute}.json`
✅ **No changes needed to:** Run files, data loader, result merging
✅ **Dynamic prompts fallback:** If dynamic prompt fails, uses vanilla as fallback

