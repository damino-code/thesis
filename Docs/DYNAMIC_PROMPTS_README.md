# Dynamic Prompts System

## Overview
This folder contains documentation for the **Dynamic Prompts System** which creates comment-specific annotator prompts by injecting actual annotator demographics from the dataset.

**Key Principle:** Instead of using static pre-written personas, the system dynamically builds prompts using real annotator characteristics (gender, age, race, religion, ideology) extracted from `selected_comments.csv` based on each comment's `comment_id`.

---

## Architecture

### Data Flow
```
Comment ID (from dataset)
    ↓
annotator_features_loader.py
    ├─ Lookup comment_id in selected_comments.csv
    └─ Extract annotator features: {gender, age, race, religion, ideology}
    ↓
single_attribute_analyzer.py._build_dynamic_persona()
    ├─ Receive features
    └─ Build system message
    ↓
LLM receives dynamic system prompt + base attribute prompt
    ↓
LLM provides prediction from this annotator's perspective
```

---

## Example Output

### Example 1: Conservative Old White Christian Woman
**Comment ID:** 39073
**Annotator Features from CSV:**
```json
{
  "comment_id": 39073,
  "annotator_gender": "Women",
  "annotator_age": 81.0,
  "age_category": "old",
  "annotator_race": "White",
  "annotator_religion": "Christian",
  "annotator_ideology": "Conservative"
}
```

**System Prompt Generated:**
```
You are a content annotator with the following characteristics:
- Gender: Women
- Age: 81.0
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator.
```

**Full Prompt Sent to LLM:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a content annotator with the following characteristics:
- Gender: Women
- Age: 81.0
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator.

<|eot_id|><|start_header_id|>user<|end_header_id|>

SCORING CRITERIA (Answer based on the definitions below):
- SENTIMENT (0-4): Question: How would you describe the sentiment? 0=Strongly negative, 1=Somewhat negative, 2=Neutral, 3=Somewhat positive, 4=Strongly positive

Analyze this comment for SENTIMENT:
Comment: "Afghanistan will rise. Kashmir will rise. Palestine will rise..."

RESPOND WITH ONLY THE NUMBER (0-4).

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

**LLM Output:** `0` (This old woman, as a Conservative Christian, perceives the statement as strongly negative)

---

### Example 2: Slightly Liberal Young Asian Christian Male (Different Perspective)
**Comment ID:** 20012
**Annotator Features from CSV:**
```json
{
  "comment_id": 20012,
  "annotator_gender": "Men",
  "annotator_age": 40.0,
  "age_category": "young",
  "annotator_race": "Asian",
  "annotator_religion": "Christian",
  "annotator_ideology": "Slightly Liberal"
}
```

**System Prompt Generated:**
```
You are a content annotator with the following characteristics:
- Gender: Men
- Age: 40.0
- Race: Asian
- Religion: Christian
- Ideology: Slightly Liberal

Evaluate this comment from your perspective as this annotator.
```

**Same Comment Text:** "Afghanistan will rise. Kashmir will rise. Palestine will rise..."

**LLM Output:** `2` (This young man, as Slightly Liberal Asian, perceives the statement more neutrally)

---

### Example 3: Old Secular Woman (Missing Ideology)
**Comment ID:** 20067
**Annotator Features from CSV:**
```json
{
  "comment_id": 20067,
  "annotator_gender": "Women",
  "annotator_age": 72.0,
  "age_category": "old",
  "annotator_race": "White",
  "annotator_religion": "Nothing",
  "annotator_ideology": "Not specified"  // ← Missing/NaN handled gracefully
}
```

**System Prompt Generated:**
```
You are a content annotator with the following characteristics:
- Gender: Women
- Age: 72.0
- Race: White
- Religion: Nothing
- Ideology: Not specified

Evaluate this comment from your perspective as this annotator.
```

**Note:** "Not specified" entries are handled gracefully without errors.

---

## Key Features

### ✅ Per-Comment Customization
Instead of one persona per run, EACH comment gets a prompt customized to its original annotator's characteristics.

### ✅ Real Demographics
Demographics are pulled directly from the dataset - NOT fictional personas.

### ✅ Robust Error Handling
- Missing values → "Not specified"
- Comment not found → Falls back to standard prompt
- NaN values → Properly handled

### ✅ Caching for Performance
Features are cached after first lookup to avoid repeated CSV queries.

---

## Implementation Files

1. **annotator_features_loader.py**
   - Loads CSV
   - Creates comment_id → features lookup
   - Provides `get_features(comment_id)` method

2. **single_attribute_analyzer.py**
   - `__init__(llm_model, use_dynamic=False)` → Conditionally initializes loader
   - `_build_dynamic_persona(comment_id)` → Builds system message
   - `analyze_attribute(text, attribute, comment_id)` → Injects persona

3. **shared_runner.py**
   - Passes `use_dynamic` and `comment_id` through pipeline
   - Saves results to `persona_results/` folder when in dynamic mode

4. **merge_results.py**
   - Accepts `use_dynamic` parameter
   - Merges from correct folder

5. **evaluation.py**
   - User selects which results to evaluate (Vanilla vs Persona)

---

## Usage

### Run Vanilla Mode (Static Prompts)
```bash
python src/run_all_attributes.py
# Select: 1 (Vanilla)
```

### Run Feature Mode (Dynamic Prompts)
```bash
python src/run_all_attributes.py
# Select: 2 (Feature)
```

---

## Result Organization

### Vanilla Mode Results
```
/results/
├─ single_attribute_analyser/
│  ├─ sentiment/results_sentiment_*.csv
│  └─ ...
└─ merged_standard_results_*.csv
```

### Feature Mode Results
```
/results/
└─ persona_results/
   ├─ sentiment/results_sentiment_*.csv
   └─ ...
   └─ merged_persona_results_*.csv
```

**This folder separation allows easy comparison between vanilla and feature-based results.**

---

## Evaluation

When running evaluation:
```bash
python src/evaluation.py
# Select: 1 (Standard) or 2 (Persona)
```

The system loads the appropriate merged results and compares LLM predictions against human annotations, showing:
- Correlation per attribute
- MAE (Mean Absolute Error)
- Classification metrics for hate speech
- Visualizations

---

## Research Value

This system enables:
1. **Annotator Bias Detection** - How do different demographic groups perceive comments?
2. **Model Perspective-Taking** - Can the LLM adopt different viewpoints?
3. **Fairness Analysis** - Do predictions vary significantly by demographic context?
4. **Comparative Evaluation** - Vanilla vs Feature-based performance differences

---

## Technical Notes

- Features are extracted once and cached (no repeated CSV reads)
- Handles up to 45 selected comments with minimal overhead
- Gracefully falls back to standard prompt if comment not in CSV
- NaN/missing values display as "Not specified" instead of errors
