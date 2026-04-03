# 📚 Dynamic Prompts Documentation - Complete Guide

## Where to Find Everything

### 📖 For Understanding the System
1. **README.md** ← Start here
   - Overview and navigation guide
   - Quick start by use case
   - Links to all other docs

### 🎯 For Examples
2. **example_dynamic_prompts.json** ← See working examples
   - 5 complete examples with:
     - Real comment_id values
     - Annotator metadata (gender, age, **age_category**, race, religion, ideology)
     - System prompts that will be generated
     - Scoring criteria for each attribute
     - Expected LLM outputs
     - Research implications
   
   **Example 1:** Conservative **old** (81.0 > 50) White Christian Woman
   - Comment: "Afghanistan will rise..."
   - Attribute: SENTIMENT (0-4 scale)
   - Prediction: 0 (Strongly negative)
   
   **Example 2:** Slightly Liberal **young** (40.0 ≤ 50) Asian Christian Male
   - Same comment as Example 1
   - Prediction: 2 (Neutral) ← Different age category, different prediction!
   
   **Example 4 vs 5:** Same hate speech comment, different demographics
   - Young Black Liberal Woman: Predicts 0 (yes, hate speech)
   - Old Conservative White Man: Predicts 1-2 (No/Unclear)
   - **Shows demographic bias in fairness!**

### 🔧 For Code Snippets
3. **CODE_SNIPPETS.md** ← See exact code
   - **Feature Loading Code** (annotator_features_loader.py lines 28-46)
     - Shows how features are extracted from CSV
     - Shows how age_category is calculated (old if > 50, young if ≤ 50)
     - Shows caching mechanism
   
   - **Dynamic Persona Building** (single_attribute_analyzer.py lines 46-62)
     - Shows how system prompt is built with features
     - Shows exact template format
   
   - **Per-Feature Analysis** (single_attribute_analyzer.py lines 64-95)
     - Shows how each attribute uses its scoring criteria
     - Shows how dynamic vs vanilla prompts differ
   
   - **Feature-Specific Examples for Each Attribute:**
     - SENTIMENT (0-4): Strongly negative → Strongly positive
     - HATESPEECH (0-2): yes / No / Unclear
     - RESPECT (0-4): Strongly disrespectful → Strongly respectful
     - DEHUMANIZE (0-4): Strongly no → Strongly yes
     - ATTACK_DEFEND (0-4): Strongly defending → Strongly attacking
     - VIOLENCE (0-4): Strongly no → Strongly yes
     - GENOCIDE (0-4): Strongly no → Strongly yes

### 📊 For Vanilla vs Dynamic Comparison
4. **VANILLA_VS_DYNAMIC.md**
   - Side-by-side comparison of modes
   - Real examples with actual outputs
   - Why the system matters for research
   - File organization after running both modes

### 📝 For Full Overview
5. **DYNAMIC_PROMPTS_README.md**
   - System architecture
   - 3 example outputs
   - Implementation files overview
   - Usage instructions
   - Result organization

### 🛠️ For Implementation Details
6. **IMPLEMENTATION_GUIDE.md**
   - Complete code walkthrough
   - All 5 key files explained
   - Data flow diagrams
   - Performance considerations
   - Error handling

---

## 🎓 Quick Learning Path

### If you want to... → Go to:

| Objective | Document | Key Section |
|-----------|----------|------------|
| **Understand what the system does** | README.md | Overview |
| **See working examples** | example_dynamic_prompts.json | All 5 examples |
| **Learn about age categories** | CODE_SNIPPETS.md | Age Category Calculation (lines 39-43) |
| **See exact code for features** | CODE_SNIPPETS.md | Feature Loading Code (lines 28-46) |
| **See exact code for personas** | CODE_SNIPPETS.md | Dynamic Persona Building (lines 46-62) |
| **Compare vanilla vs dynamic** | VANILLA_VS_DYNAMIC.md | Full Example section |
| **Understand fairness implications** | example_dynamic_prompts.json | Examples 4 vs 5 |
| **See how each attribute works** | CODE_SNIPPETS.md | Feature-Specific Examples |
| **Trace code through system** | CODE_SNIPPETS.md | How to Trace Dynamic Prompts |
| **Find exact line numbers** | CODE_SNIPPETS.md | Key Code Locations Summary |

---

## 🔍 Age Category Logic (The Key Detail!)

**File:** `annotator_features_loader.py` (Lines 39-43)

```python
# Determine age category: old if > 50, young if ≤ 50
try:
    age_value = float(features['age']) if features['age'] != 'Not specified' else None
    if age_value is not None:
        features['age_category'] = 'old' if age_value > 50 else 'young'
    else:
        features['age_category'] = 'Not specified'
```

**Examples in actual data:**
- 81.0 → **old** (> 50)
- 40.0 → **young** (≤ 50) 
- 72.0 → **old** (> 50)
- 25.0 → **young** (≤ 50)
- 65.0 → **old** (> 50)

---

## 📋 All 5 Examples at a Glance

| Ex | Age | Category | Gender | Religion | Ideology | Attribute | Prediction |
|----|-----|----------|--------|----------|----------|-----------|-----------|
| 1 | 81.0 | **old** | Woman | Christian | Conservative | SENTIMENT | 0 |
| 2 | 40.0 | **young** | Man | Christian | Liberal | SENTIMENT | 2 |
| 3 | 72.0 | **old** | Woman | Nothing | Not specified | SENTIMENT | 2-3 |
| 4 | 25.0 | **young** | Woman | Christian | Liberal | HATESPEECH | 0 |
| 5 | 65.0 | **old** | Man | Christian | Conservative | HATESPEECH | 1-2 |

**Key insight:** Examples 1 vs 2 (same comment, different ages) and 4 vs 5 (same comment, different demographics) show how predictions differ!

---

## 🚀 How to Use This Guide

### Step 1: Understand the Concept
- Read: **README.md** (2-3 min)

### Step 2: See It In Action
- Read: **example_dynamic_prompts.json** (5 min)
- Note the age_category field - it's automatically calculated!
- Compare Example 1 (old) vs Example 2 (young) vs Example 4 vs 5

### Step 3: See the Code
- Read: **CODE_SNIPPETS.md** sections:
  - "Feature Loading Code" (shows age_category calculation)
  - "Dynamic Persona Building" (shows prompt construction)
  - "Feature-Specific Examples" (shows each attribute's scoring)

### Step 4: Deep Dive (Optional)
- Read: **IMPLEMENTATION_GUIDE.md** for complete technical walkthrough
- Read: **VANILLA_VS_DYNAMIC.md** for research implications

---

## ✅ Updated Elements

✅ All examples include `age_category` field
✅ Age category logic: > 50 = old, ≤ 50 = young
✅ All scoring criteria match your actual prompt JSON files
✅ Code snippets show exact lines and file locations
✅ Feature loading code produces age_category automatically
✅ Dynamic persona building creates system messages with all features
✅ Per-attribute analysis shows how each of 7 attributes work

---

## 🎯 Where is what?

### Code for Feature Calculation
- **File:** `annotator_features_loader.py`
- **Method:** `get_features(comment_id)`
- **Lines:** 28-46
- **Key Line:** 42 uses `'old' if age_value > 50 else 'young'`

### Code for Dynamic Prompts
- **File:** `single_attribute_analyzer.py`
- **Method:** `_build_dynamic_persona(comment_id)`
- **Lines:** 46-62
- **Output:** System message with all 5 features + age_category

### Code for Per-Attribute Analysis
- **File:** `single_attribute_analyzer.py`
- **Method:** `analyze_attribute(text, attribute, comment_id)`
- **Lines:** 64-95
- **Key:** Loads scoring criteria from `prompts/{attribute}.json`

### Scoring Criteria for Each Attribute
- **Files:** `prompts/sentiment.json`, `prompts/hatespeech.json`, etc.
- **Location:** `/src/prompts/` folder
- **Format:** JSON with exact scoring scales and definitions

