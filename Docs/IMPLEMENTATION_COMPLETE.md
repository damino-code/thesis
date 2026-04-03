# ✅ Implementation Complete: Attribute-Specific Dynamic Prompts

## What Changed

### Before (Generic Dynamic Prompts)
```
_build_dynamic_persona(comment_id)
    ├─ Always same system message
    └─ Get features, build generic persona
        
All attributes get: 
"You are a content annotator with characteristics: {features}"
+ Standard prompt from prompts/{attribute}.json
```

### After (Attribute-Specific Dynamic Prompts)
```
_build_dynamic_prompt(attribute, comment_id, text)
    ├─ Load: prompts/dynamic/{attribute}.json (DIFFERENT for each attribute!)
    ├─ Get features from CSV
    ├─ Format template with features + text
    └─ Return complete personalized prompt

Each attribute gets its own:
"You are a content annotator with characteristics: {features}
...attribute-specific scoring criteria..."
```

---

## New Files Created

### 7 Dynamic Prompt Templates
```
✅ src/prompts/dynamic/hatespeech.json
✅ src/prompts/dynamic/sentiment.json
✅ src/prompts/dynamic/respect.json
✅ src/prompts/dynamic/dehumanize.json
✅ src/prompts/dynamic/attack_defend.json
✅ src/prompts/dynamic/violence.json
✅ src/prompts/dynamic/genocide.json
```

Each includes:
- Persona with feature placeholders: {gender}, {age}, {age_category}, {race}, {religion}, {ideology}
- Attribute-specific scoring criteria
- Comment text placeholder: {text}

### Documentation
✅ `src/persona_prompts/ATTRIBUTE_SPECIFIC_DYNAMIC_PROMPTS.md`

---

## Code Changes in `single_attribute_analyzer.py`

### 1. Constructor Updated (Line 14)
**Before:**
```python
self.prompts = self._load_prompts()
```

**After:**
```python
self.prompts = self._load_prompts()
self.dynamic_prompts = self._load_dynamic_prompts() if use_dynamic else {}
```

Loads all 7 attribute-specific dynamic prompt templates on startup when `use_dynamic=True`

---

### 2. New Method: `_load_dynamic_prompts()` (Lines 48-68)
```python
def _load_dynamic_prompts(self):
    """Load attribute-specific dynamic prompt templates"""
    dynamic_prompts = {}
    dynamic_prompt_dir = os.path.join(..., 'prompts', 'dynamic')
    
    # Load from prompts/dynamic/*.json
    for filename in os.listdir(dynamic_prompt_dir):
        if filename.endswith(".json"):
            attribute = os.path.splitext(filename)[0]
            # Each file has 'dynamic_prompt' key (not 'prompt')
            dynamic_prompts[attribute] = formatted_template
    
    return dynamic_prompts
```

---

### 3. New Method: `_build_dynamic_prompt()` (Lines 102-131)
```python
def _build_dynamic_prompt(self, attribute, comment_id, text):
    """Build complete dynamic prompt with features injected for a specific attribute"""
    
    # Get features from CSV
    features = self.features_loader.get_features(comment_id)
    
    # Load template for THIS SPECIFIC ATTRIBUTE
    dynamic_template = self.dynamic_prompts[attribute]
    
    # Format with features + text
    formatted_prompt = dynamic_template.format(
        gender=features['gender'],
        age=features['age'],
        age_category=features['age_category'],  # old or young!
        race=features['race'],
        religion=features['religion'],
        ideology=features['ideology'],
        text=text[:500]
    )
    
    return formatted_prompt
```

---

### 4. Updated: `analyze_attribute()` (Lines 133-180)
**Before:**
```python
if self.use_dynamic:
    persona_prefix = self._build_dynamic_persona(comment_id)
else:
    persona_prefix = "You are an expert content annotator."

full_prompt = f"{persona_prefix}{standard_prompt}"
```

**After:**
```python
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
```

---

## How It Works Now

### Step 1: User Selects Dynamic Mode
```bash
python src/run_all_attributes.py
→ Select: 2 (Feature mode)
```

### Step 2: System Initializes
```
SingleAttributeAnalyzer(llm_model, use_dynamic=True)
├─ _load_prompts() → loads standard prompts
└─ _load_dynamic_prompts() → loads all 7 dynamic prompt templates
    ├─ Loads dynamic/hatespeech.json
    ├─ Loads dynamic/sentiment.json
    ├─ Loads dynamic/respect.json
    ├─ ... (all 7 templates)
    └─ ✅ 7 templates loaded
```

### Step 3: Analyze Comment for SENTIMENT
```
analyze_attribute(text="...", attribute="sentiment", comment_id=39073)
    └─ _build_dynamic_prompt("sentiment", 39073, text)
        ├─ Get features for comment_id 39073
        │   └─ {gender: Women, age: 81.0, age_category: old, ...}
        ├─ Load prompts/dynamic/sentiment.json template
        ├─ Format with features + text
        └─ Return complete personalized prompt
```

### Step 4: Analyze Same Comment for HATESPEECH
```
analyze_attribute(text="...", attribute="hatespeech", comment_id=39073)
    └─ _build_dynamic_prompt("hatespeech", 39073, text)
        ├─ Get features for comment_id 39073 (SAME features)
        ├─ Load prompts/dynamic/hatespeech.json template (DIFFERENT!)
        ├─ Format with features + text
        └─ Return complete personalized prompt
```

**Key Difference:** 
- Same annotator (81-year-old Conservative woman)
- Same comment  
- **Different prompt templates** (sentiment vs hatespeech)
- Each attribute gets its own personalized evaluation!

---

## Example Outputs

### SENTIMENT Dynamic Prompt
```
You are a content annotator with the following characteristics:
- Gender: Women
- Age: 81.0 (old)
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator.

SCORING CRITERIA (Answer based on the definitions below):
- SENTIMENT (0-4): Question: How would you describe the sentiment? 
  0=Strongly negative, 1=Somewhat negative, 2=Neutral, 3=Somewhat positive, 4=Strongly positive

Analyze this comment for SENTIMENT:
Comment: "Afghanistan will rise. Kashmir will rise..."

RESPOND WITH ONLY THE NUMBER (0-4).
```

### HATESPEECH Dynamic Prompt (Same annotator, same comment)
```
You are a content annotator with the following characteristics:
- Gender: Women
- Age: 81.0 (old)
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator.

SCORING CRITERIA (Answer based on the definitions below):
- HATESPEECH (0-2): Question: Does this comment contain hate speech? 
  0=yes , 1=No, 2=Unclear/Neutral

Analyze this comment for HATE SPEECH:
Comment: "Afghanistan will rise. Kashmir will rise..."

RESPOND WITH ONLY THE NUMBER (0-2).
```

---

## Architecture Comparison

| Aspect | Before | After |
|--------|--------|-------|
| **System Prompt** | Generic "You are a content annotator" | Personalized with features |
| **Per Attribute** | Same persona for all attributes | Unique template per attribute |
| **Files** | Standard prompts only | Standard + Dynamic prompts |
| **Method** | `_build_dynamic_persona()` | `_build_dynamic_prompt()` |
| **Personalization** | Shallow (just mentions features) | Deep (features + attribute criteria) |
| **Fairness Analysis** | Limited | Enhanced (see differences per attribute!) |

---

## Benefits

✅ **Each attribute has its own dynamic "voice"**
- SENTIMENT: "How would YOU describe the sentiment?"
- HATESPEECH: "Does THIS comment contain hate speech?"
- RESPECT: "Is comment respectful to YOUR group?"

✅ **More authentic perspective-taking**
- LLM personalizes evaluation to the annotator
- Not just adding features, but integrating them into evaluation criteria

✅ **Better research insights**
- Can see how age_category (old/young) affects EACH attribute
- Can compare fairness across different attributes for same annotator

✅ **Scalable**
- Easy to add new attributes: just create `prompts/dynamic/{attribute}.json`
- Same architecture for all 7 attributes

---

## Testing

### Verify Files
```bash
ls -la src/prompts/dynamic/
# Should show 7 JSON files
```

### Verify Loading
```bash
python src/run_all_attributes.py
# Output should show:
# ✅ Loaded 7 dynamic prompt templates
```

### Verify Output
```python
from single_attribute_analyzer import SingleAttributeAnalyzer

analyzer = SingleAttributeAnalyzer(model, use_dynamic=True)
# Should see: "✅ Loaded 7 dynamic prompt templates"

prompt = analyzer._build_dynamic_prompt("sentiment", 39073, "test comment")
# Should see persona + sentiment-specific criteria
```

---

## Key Takeaway

❌ **Old Way:** "Generic persona → + standard prompt"
✅ **New Way:** "Attribute-specific template with integrated persona"

This makes the dynamic prompts truly **personalized per attribute**, not just generic persona + standard prompt!

