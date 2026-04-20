import re
import math
import os
import json
from annotator_features_loader import AnnotatorFeaturesLoader


def softmax_dict(score_dict):
    """Apply softmax over a dict of {label: logprob}."""
    m = max(score_dict.values())
    exps = {k: math.exp(v - m) for k, v in score_dict.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def confidence_from_label_logprobs(label_logprobs):
    """Compute confidence and predicted label from label logprobs."""
    probs = softmax_dict(label_logprobs)
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    pred_label, conf = ranked[0]

    return {
        "label": pred_label,
        "confidence": conf,
    }


class SingleAttributeAnalyzer:
    def __init__(self, llm_model, use_dynamic=False):
        self.model = llm_model
        self.use_dynamic = use_dynamic
        self.prompts = {}
        self.labels = {}
        self._load_prompts()
        self.dynamic_prompts = self._load_dynamic_prompts() if use_dynamic else {}

        # Load feature loader only if dynamic mode is enabled
        if self.use_dynamic:
            self.features_loader = AnnotatorFeaturesLoader()
            self.features_loader.load_features()
            print("📊 Using Feature-based dynamic prompts")
            print(f"✅ Loaded {len(self.dynamic_prompts)} dynamic prompt templates")
        else:
            self.features_loader = None
            print("📄 Using Vanilla standard prompts")

    def _load_prompts(self):
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        if not os.path.exists(prompt_dir):
            return
        for filename in os.listdir(prompt_dir):
            if filename.endswith(".json"):
                attribute = os.path.splitext(filename)[0]
                with open(os.path.join(prompt_dir, filename), "r") as handle:
                    data = json.load(handle)
                    if isinstance(data.get("prompt"), list):
                        self.prompts[attribute] = "\n".join(data["prompt"])
                    else:
                        self.prompts[attribute] = data.get("prompt", "")
                    self.labels[attribute] = data.get('labels', ["0", "1", "2", "3", "4"])

    def _load_dynamic_prompts(self):
        """Load attribute-specific dynamic prompt templates"""
        dynamic_prompts = {}
        dynamic_prompt_dir = os.path.join(os.path.dirname(__file__), 'persona_prompts')
        print("📂 Loading dynamic prompt templates...")

        if not os.path.exists(dynamic_prompt_dir):
            print(f"⚠️  Dynamic prompts folder not found: {dynamic_prompt_dir}")
            return dynamic_prompts

        for filename in os.listdir(dynamic_prompt_dir):
            if filename.startswith('dynamic_') and filename.endswith(".json"):
                base_name = os.path.splitext(filename)[0]
                attribute = base_name.replace('dynamic_', '')

                with open(os.path.join(dynamic_prompt_dir, filename), 'r') as f:
                    data = json.load(f)
                    if isinstance(data['dynamic_prompt'], list):
                        dynamic_prompts[attribute] = '\n'.join(data['dynamic_prompt'])
                    else:
                        dynamic_prompts[attribute] = data['dynamic_prompt']
                    if 'labels' in data:
                        self.labels[attribute] = data['labels']
                    print(f"  ✅ Loaded dynamic prompt for: {attribute}")

        print(f"📊 Total dynamic prompts loaded: {len(dynamic_prompts)}")
        return dynamic_prompts

    def _build_dynamic_prompt(self, attribute, comment_id, text, annotator_id=None):
        """Build dynamic prompt with annotator features injected.

        annotator_id must be passed so the correct annotator's demographics are
        used. Without it, get_features() falls back to the first matching row,
        which will be wrong whenever a comment has multiple annotators.
        """
        if attribute not in self.dynamic_prompts:
            print(f"❌ Attribute '{attribute}' not in dynamic_prompts. Available: {list(self.dynamic_prompts.keys())}")
            return None

        features = self.features_loader.get_features(comment_id, annotator_id)
        if features is None:
            print(f"❌ No features found for comment_id={comment_id}, annotator_id={annotator_id}")
            return None

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
            print(f"❌ Missing key in features: {e}")
            print(f"   Available features: {list(features.keys())}")
            return None
        except Exception as e:
            print(f"❌ Error formatting dynamic prompt: {e}")
            return None

    def _extract_label_logprobs(self, choice, valid_labels):
        """Extract logprobs for valid label tokens from a llama-cpp-python response choice."""
        label_logprobs = {}
        if 'logprobs' in choice and choice['logprobs'] and 'top_logprobs' in choice['logprobs']:
            top_logprobs = choice['logprobs']['top_logprobs']
            if top_logprobs and top_logprobs[0]:
                for token, logprob in top_logprobs[0].items():
                    token_stripped = token.strip()
                    if token_stripped in valid_labels:
                        label_logprobs[token_stripped] = logprob
        return label_logprobs

    def analyze_attribute(self, text, attribute, comment_id=None, annotator_id=None):
        """Analyze a comment for a single attribute"""
        if attribute not in self.prompts:
            raise ValueError(f"Unknown attribute: {attribute}")

        valid_labels = self.labels.get(attribute, ["0", "1", "2", "3", "4"])

        # Build prompt content
        if self.use_dynamic and comment_id is not None:
            user_message = self._build_dynamic_prompt(attribute, comment_id, text, annotator_id)
            if user_message is None:
                user_message = self.prompts[attribute].format(text=text[:500])
                system_content = "You are an expert content moderator."
            else:
                system_content = "You are an expert content annotator responding from the perspective defined in the following instructions."
        else:
            system_content = "You are an expert content moderator."
            user_message = self.prompts[attribute].format(text=text[:500])

        full_prompt = f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

        try:
            # First attempt with logprobs=10
            response = self.model.create_completion(
                prompt=full_prompt,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                logprobs=10,
                echo=False,
            )

            choice = response['choices'][0]
            label_logprobs = self._extract_label_logprobs(choice, valid_labels)

            if label_logprobs and len(label_logprobs) == len(valid_labels):
                result = confidence_from_label_logprobs(label_logprobs)
                return {
                    attribute: float(result["label"]),
                    'confidence': result["confidence"],
                }

            # Retry with logprobs=20
            response = self.model.create_completion(
                prompt=full_prompt,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                logprobs=20,
                echo=False,
            )

            choice = response['choices'][0]
            label_logprobs = self._extract_label_logprobs(choice, valid_labels)

            if label_logprobs and len(label_logprobs) == len(valid_labels):
                result = confidence_from_label_logprobs(label_logprobs)
                return {
                    attribute: float(result["label"]),
                    'confidence': result["confidence"],
                }
            else:
                return {attribute: "invalid", 'confidence': 0.0}

        except Exception as e:
            print(f"❌ LLM Error in analyze_attribute({attribute}): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {attribute: "invalid", 'confidence': 0.0}
