# Dynamic Prompts System - Documentation Index

## Welcome to the Dynamic Prompts System!

This folder contains comprehensive documentation for how the dynamic prompts system injects annotator demographics into LLM evaluation prompts.

---

## 📚 Documentation Files

### 1. **DYNAMIC_PROMPTS_README.md** ← START HERE
**Purpose:** Overview of the whole system

**Contains:**
- System architecture and data flow diagram
- 3 complete example outputs showing how prompts are injected
- Key features and implementation files overview
- Usage instructions (Vanilla vs Feature mode)
- Result organization structure
- Research implications

**Read this first for:** Understanding what the system does and why it matters

---

### 2. **VANILLA_VS_DYNAMIC.md**
**Purpose:** Direct comparison between static and dynamic prompting

**Contains:**
- Side-by-side comparison of vanilla vs dynamic modes
- Full example: Same comment evaluated by different annotators
- Key insights and research value
- Implementation comparison
- CSV data example showing how same comment has multiple annotators
- File organization after running both modes
- Evaluation framework differences

**Read this for:** Understanding the concrete differences in prompting strategy and research implications

---

### 3. **example_dynamic_prompts.json**
**Purpose:** Machine-readable examples with detailed metadata

**Contains:**
- 5 JSON example objects with real scenarios:
  - Example 1: Conservative elderly White Christian woman
  - Example 2: Slightly Liberal Asian male (same comment)
  - Example 3: Secular woman with missing ideology
  - Example 4: Hate speech from young Black liberal woman's perspective
  - Example 5: Same hate speech from Conservative White man's perspective
- Implementation details for feature extraction
- Research questions the system addresses

**Read this for:** Concrete JSON format examples and machine parsing of prompt structure

---

### 4. **IMPLEMENTATION_GUIDE.md**
**Purpose:** Code-level explanation of how the system works

**Contains:**
- Architecture diagram with data flow
- Code walkthrough for all 5 key files:
  - run_all_attributes.py (user mode selection)
  - annotator_features_loader.py (CSV loading + caching)
  - single_attribute_analyzer.py (dynamic persona building)
  - shared_runner.py (result folder organization)
  - merge_results.py (dual-mode merging)
  - evaluation.py (mode-aware evaluation)
- Step-by-step data flow example
- Performance considerations
- Error handling strategies
- Testing code snippets
- Verification commands

**Read this for:** Understanding the Python implementation and how features flow through code

---

## 🎯 Quick Start by Use Case

### "I want to understand what the system does"
1. Read: **DYNAMIC_PROMPTS_README.md**
2. Look at: **VANILLA_VS_DYNAMIC.md** - see real examples
3. Skim: **example_dynamic_prompts.json** - see JSON structure

### "I need to run the system"
1. Read: **DYNAMIC_PROMPTS_README.md** - understand modes
2. Follow: Usage instructions in DYNAMIC_PROMPTS_README.md
3. Check: Result organization to find your outputs

### "I'm debugging or extending the code"
1. Start with: **IMPLEMENTATION_GUIDE.md**
2. Reference: Code walkthrough sections
3. Use: Verification commands to test

### "I want to understand the research value"
1. Read: **VANILLA_VS_DYNAMIC.md** - research questions section
2. Study: example_dynamic_prompts.json - see disparities in Example 4 vs 5
3. Review: "Key Insights" table in VANILLA_VS_DYNAMIC.md

---

## 📊 System Overview at a Glance

**What it does:**
- Creates unique LLM evaluation prompts per comment based on the annotator who originally reviewed it
- Injects real demographic features (gender, age, race, religion, ideology) from selected_comments.csv
- Enables LLM perspective-taking: "What would this 81-year-old Conservative Christian woman evaluate this as?"

**Why it matters:**
- Quantify demographic bias in content moderation
- Test if LLM can adopt different demographic perspectives
- Compare vanilla (neutral) vs feature-based (perspective) evaluation patterns

**How it works:**
1. User selects mode: Vanilla (1) or Feature (2)
2. System loads annotator features from CSV (cached efficiently)
3. For each comment, builds dynamic system prompt with demographics
4. LLM evaluates with demographic context
5. Results saved to separate folders (vanilla vs persona)
6. Evaluation reports show differences between modes

---

## 🔄 Data Flow

```
selected_comments.csv (annotator demographics)
         ↓
Feature Loader (caches in memory)
         ↓
Single Analyzer (builds dynamic persona)
         ↓
LLM receives: "You are {demographics}..."
         ↓
Results: persona_results/ (feature) or single_attribute_analyser/ (vanilla)
         ↓
Evaluation & Visualization
```

---

## 📁 Related Files in Codebase

These files implement the system:
- `annotator_features_loader.py` - CSV loading and caching
- `single_attribute_analyzer.py` - Dynamic prompt building
- `shared_runner.py` - Orchestration with conditional folders
- `run_all_attributes.py` - User interface and mode selection
- `merge_results.py` - Combines results by mode
- `evaluation.py` - Mode-aware evaluation

All in: `/storage/home/amine/thesis/Llama-3.2-8B-Instruct-Q4_K_M/separate_hatespeech_multiattribute/src/`

---

## 🧪 Key Examples from Documentation

### Example 1: Conservative Woman
- Comment: "Afghanistan will rise. Kashmir will rise..."
- Annotator: 81-year-old Conservative Christian White woman
- System Prompt: "You are a woman, age 81, White, Christian, Conservative..."
- Prediction: 2 (Negative - interprets as concerning from her perspective)

### Example 2: Liberal Man (Same Comment)
- Comment: Same as above
- Annotator: 80-year-old Slightly Liberal Asian Christian man
- System Prompt: "You are a man, age 80, Asian, Christian, Slightly Liberal..."
- Prediction: 1 (Neutral/Mixed - interprets as aspirational from his perspective)

**Key Insight:** Same comment, same attribute, different demographics → different predictions = FAIRNESS DISPARITY

---

## ⚙️ System Features

✅ Per-comment customization (not per-run)
✅ Real demographic data (from CSV, not fictional)
✅ Efficient caching (avoids repeated CSV reads)
✅ Graceful error handling (missing values → "Not specified")
✅ Clean mode toggle (vanilla=1, feature=2)
✅ Result separation (no mode mixing)
✅ Evaluation flexibility (evaluate either mode)

---

## 🔍 Research Questions Answered

The system enables research into:
1. How demographic identity influences hate speech detection
2. Does LLM exhibit perspectival reasoning based on demographics?
3. Are there fairness disparities across different identity groups?
4. Can LLM predictions align with human evaluators from same demographic?
5. What role does annotator identity play in content moderation?

---

## 💡 Quick Tips

- **Vanilla results** go to: `results/single_attribute_analyser/`
- **Feature results** go to: `results/persona_results/`
- **Always clear results before re-running** same mode to avoid confusion
- **Compare merged results** to see biggest disparities between modes:
  - `cat results/merged_standard_results_*.csv`
  - `cat results/merged_persona_results_*.csv`
- **Use evaluation.py twice** - once for Vanilla (1), once for Persona (2), then compare metrics

---

## 📞 Need Help?

- **System doesn't run?** → Check IMPLEMENTATION_GUIDE.md verification commands
- **Results look wrong?** → Check merged files are from correct mode
- **Don't understand a concept?** → VANILLA_VS_DYNAMIC.md has best explanations
- **Want to extend code?** → IMPLEMENTATION_GUIDE.md shows architecture

---

**Last Updated:** January 2025
**System Version:** 1.0 (Dynamic Prompts with Feature Injection)
**Status:** ✅ Fully Implemented and Documented
