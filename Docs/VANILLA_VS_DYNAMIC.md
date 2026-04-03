# Vanilla vs Dynamic Prompts: Side-by-Side Comparison

## What's the Difference?

### VANILLA MODE (Standard Static Prompt)
Every comment is evaluated with the **same generic system prompt**:

```
You are a content moderator. Evaluate this comment objectively.
```

**Result:** All comments evaluated from a neutral third-party perspective - no demographic context.

---

### DYNAMIC MODE (Feature-Injected Prompt)
Each comment is evaluated with a **unique annotator-specific system prompt**:

```
You are a content moderator with the following characteristics:
- Gender: Women
- Age: 81.0
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator.
```

**Result:** Comments evaluated from the lens of their original annotator's demographic identity.

---

## Full Example: Same Comment, Two Modes

**Comment:** "The UN should enforce international law on countries that violate human rights"

**Comment ID:** 42501

---

## ① VANILLA MODE Evaluation

### System Prompt
```
You are a content moderator. Evaluate this comment objectively.
```

### User Prompt
```
SCORING CRITERIA (Answer based on the definitions below):
- HATESPEECH (0-2): Question: Does this comment contain hate speech? 0=yes , 1=No, 2=Unclear/Neutral

Analyze this comment for HATE SPEECH:
Comment: "The UN should enforce international law on countries that violate human rights"

RESPOND WITH ONLY THE NUMBER (0-2).
```

### LLM Response
```
0
```

**Interpretation:** Neutral evaluator says this is not hate speech. It's a political/humanitarian statement.

---

## ② DYNAMIC MODE Evaluation (Liberal Female Human Rights Advocate)

### Extracted Annotator Profile
```json
{
  "comment_id": 42501,
  "annotator_gender": "Women",
  "annotator_age": 35.0,
  "age_category": "young",
  "annotator_race": "Black",
  "annotator_religion": "Atheist",
  "annotator_ideology": "Very Liberal"
}
```

### System Prompt (Dynamically Built)
```
You are a content moderator with the following characteristics:
- Gender: Women
- Age: 35.0
- Race: Black
- Religion: Atheist
- Ideology: Very Liberal

Evaluate this comment from your perspective as this annotator.
```

### User Prompt (Same as Vanilla)
```
SCORING CRITERIA (Answer based on the definitions below):
- HATESPEECH (0-2): Question: Does this comment contain hate speech? 0=yes , 1=No, 2=Unclear/Neutral

Analyze this comment for HATE SPEECH:
Comment: "The UN should enforce international law on countries that violate human rights"

RESPOND WITH ONLY THE NUMBER (0-2).
```

### LLM Response
```
0
```

**Interpretation:** Liberal young woman agrees - not hate speech, aligns with her values.

---

## ③ DYNAMIC MODE Evaluation (Conservative Nationalist Male)

### Extracted Annotator Profile
```json
{
  "comment_id": 42501,
  "annotator_gender": "Men",
  "annotator_age": 58.0,
  "age_category": "old",
  "annotator_race": "White",
  "annotator_religion": "Christian",
  "annotator_ideology": "Conservative"
}
```

### System Prompt (Dynamically Built)
```
You are a content moderator with the following characteristics:
- Gender: Men
- Age: 58.0
- Race: White
- Religion: Christian
- Ideology: Conservative

Evaluate this comment from your perspective as this annotator.
```

### User Prompt (Same)
```
SCORING CRITERIA (Answer based on the definitions below):
- HATESPEECH (0-2): Question: Does this comment contain hate speech? 0=yes , 1=No, 2=Unclear/Neutral

Analyze this comment for HATE SPEECH:
Comment: "The UN should enforce international law on countries that violate human rights"

RESPOND WITH ONLY THE NUMBER (0-2).
```

### LLM Response
```
1
```

**Interpretation:** Conservative older man may view UN intervention as overreach/threat to national sovereignty. Judges as "No" but with some ambiguity vs young woman's stronger "yes" judgment.

---

## Key Insights

| Aspect | Vanilla Mode | Dynamic Mode |
|--------|-------------|--------------|
| **System Prompt** | Generic, neutral | Annotator-specific demographics |
| **Perspective** | Third-party objective | First-person subjective |
| **Feature Context** | None | Gender, Age, Race, Religion, Ideology |
| **Same Comment, Different Annotators** | All produce 1 output | Multiple outputs per demographic |
| **Research Value** | Baseline bias measurement | Fairness analysis, perspective-taking |
| **Results Storage** | `results/single_attribute_analyser/` | `results/persona_results/` |
| **Use Case** | Model objectivity benchmark | Demographic disparities research |

---

## Why This Matters for Your Research

### Without Dynamic Prompts (Vanilla)
- You get: "LLM classified comment X as hate speech: YES/NO"
- Missing: How demographic identity influences moderation decisions
- Limitation: Can't measure fairness across demographics

### With Dynamic Prompts (Feature-Injected)
- You get: "Black liberal woman classified X as hate speech, White conservative man classified X as inflammatory"
- Gain: Clear visibility into demographic-influenced bias
- Capability: Quantify fairness disparities across identity groups

---

## Implementation Comparison

### Vanilla Pipeline
```
Comment → Analyzer (use_dynamic=False) 
    → Standard system prompt ("You are a moderator")
    → LLM prediction
    → /results/single_attribute_analyser/
```

### Dynamic Pipeline
```
Comment ID → Get Annotator Features from CSV
    ↓
Comment → Analyzer (use_dynamic=True, comment_id=X)
    → Dynamic system prompt ("You are {gender} {age} {race} {religion} {ideology}")
    → LLM prediction
    → /results/persona_results/
```

---

## CSV Data Example (selected_comments.csv)

```
comment_id,comment_text,...,annotator_gender,annotator_age,annotator_race,annotator_religion,annotator_ideology
42501,The UN should enforce...,Women,35.0,Black,Atheist,Very Liberal
42501,The UN should enforce...,Men,58.0,White,Christian,Conservative
```

**Same comment (42501) with multiple annotators having different demographics!**
- This enables the dynamic prompt system to show how each annotator (via the LLM) would evaluate it

---

## File Organization

After running both modes:

```
results/
├── single_attribute_analyser/        # Vanilla results
│   ├── sentiment/
│   ├── hate_speech/
│   └── ...attributes...
│
├── persona_results/                   # Dynamic results
│   ├── sentiment/
│   ├── hate_speech/
│   └── ...attributes...
│
├── merged_standard_results_*.csv      # Vanilla aggregated
└── merged_persona_results_*.csv       # Dynamic aggregated
```

You can now compare:
```bash
compare_results.py merged_standard_results_*.csv merged_persona_results_*.csv
```

To see **fairness disparities**: How different demographic perspectives differ in their evaluation patterns.

---

## Evaluation Framework

After both modes complete, run:

```bash
python src/evaluation.py
# Choose: 1 (Standard/Vanilla) or 2 (Persona/Dynamic)
```

**For Vanilla:**
- Compare vanilla LLM predictions vs human annotations
- Measure model objectivity

**For Dynamic:**
- Compare persona-adjusted LLM predictions vs human annotations
- Measure model's ability to match perspective-specific human judgments
- Identify demographic biases in both human and model

**Then compare the two evaluation reports** to see:
- Does the model perform better when demographically-aligned?
- Which identity groups show biggest prediction gaps?
- Is the model exhibiting "demographic alignment bias"?

