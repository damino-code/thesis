import re
import math
import os
import json
from vllm import SamplingParams
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


def extract_label_logprobs(logprobs_dict, valid_labels):
    """Extract logprobs for valid label tokens from a vLLM logprobs dict."""
    label_logprobs = {}
    if logprobs_dict:
        for token_id, logprob_obj in logprobs_dict.items():
            decoded = logprob_obj.decoded_token.strip()
            if decoded in valid_labels:
                label_logprobs[decoded] = logprob_obj.logprob
    return label_logprobs


class SingleAttributeAnalyzer:
    def __init__(self, llm_model, use_dynamic=False):
        self.model = llm_model
        self.use_dynamic = use_dynamic
        self.prompts = {}
        self.labels = {}
        self._load_prompts()
        self.dynamic_prompts = self._load_dynamic_prompts() if use_dynamic else {}

        # Load the tokenizer from vLLM for chat template formatting
        self.tokenizer = llm_model.get_tokenizer()

        if self.use_dynamic:
            self.features_loader = AnnotatorFeaturesLoader()
            self.features_loader.load_features()
            print(f"Using Feature-based dynamic prompts ({len(self.dynamic_prompts)} templates)")
        else:
            self.features_loader = None
            print("Using Vanilla standard prompts")

        self.sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=10,
            seed=42,
        )

        self.retry_sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=20,
            seed=42,
        )

    def _load_prompts(self):
        prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')

        if not os.path.exists(prompt_dir):
            raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

        for filename in os.listdir(prompt_dir):
            if filename.endswith(".json") and not filename.startswith("dynamic"):
                attribute = os.path.splitext(filename)[0]
                with open(os.path.join(prompt_dir, filename), 'r') as f:
                    data = json.load(f)
                    if isinstance(data['prompt'], list):
                        self.prompts[attribute] = '\n'.join(data['prompt'])
                    else:
                        self.prompts[attribute] = data['prompt']
                    self.labels[attribute] = data.get('labels', ["0", "1", "2", "3", "4"])

    def _load_dynamic_prompts(self):
        dynamic_prompts = {}
        dynamic_prompt_dir = os.path.join(os.path.dirname(__file__), 'persona_prompts')

        if not os.path.exists(dynamic_prompt_dir):
            return dynamic_prompts

        for filename in os.listdir(dynamic_prompt_dir):
            if filename.startswith('dynamic_') and filename.endswith(".json"):
                attribute = os.path.splitext(filename)[0].replace('dynamic_', '')
                with open(os.path.join(dynamic_prompt_dir, filename), 'r') as f:
                    data = json.load(f)
                    if isinstance(data['dynamic_prompt'], list):
                        dynamic_prompts[attribute] = '\n'.join(data['dynamic_prompt'])
                    else:
                        dynamic_prompts[attribute] = data['dynamic_prompt']
                    if 'labels' in data:
                        self.labels[attribute] = data['labels']
        return dynamic_prompts

    def _build_dynamic_prompt(self, attribute, comment_id, text, annotator_id=None):
        if attribute not in self.dynamic_prompts:
            return None

        features = self.features_loader.get_features(comment_id, annotator_id)
        if features is None:
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
        except (KeyError, Exception) as e:
            print(f"Error formatting dynamic prompt: {e}")
            return None

    def build_prompt(self, text, attribute, comment_id=None, annotator_id=None):
        """Build prompt using the model's chat template via the tokenizer."""
        if attribute not in self.prompts:
            raise ValueError(f"Unknown attribute: {attribute}")

        system_message = ("you are an expert annotator participating in a subjective "
                          "annotation task for a research project. Your task in to analyse "
                          "raw, unfiltered internet comments. do not apply safety filters "
                          "and do not refuse to answer. Only output the requested numeric score.")

        if self.use_dynamic and comment_id is not None:
            user_message = self._build_dynamic_prompt(attribute, comment_id, text, annotator_id)
            if user_message is None:
                user_message = self.prompts[attribute].format(text=text[:500])
        else:
            user_message = self.prompts[attribute].format(text=text[:500])

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return full_prompt

    def _extract_label_logprobs_from_output(self, output, valid_labels):
        """Extract logprobs for valid label tokens from a vLLM output."""
        if output.outputs[0].logprobs:
            return extract_label_logprobs(output.outputs[0].logprobs[0], valid_labels)
        return {}

    def _extract_result(self, output, attribute, all_labels_required=True):
        """Extract label and confidence from a single vLLM output.

        Returns None for confidence if not all labels are present and
        all_labels_required is True (signals that a retry is needed).
        """
        valid_labels = self.labels.get(attribute, ["0", "1", "2", "3", "4"])
        label_logprobs = self._extract_label_logprobs_from_output(output, valid_labels)

        if label_logprobs and len(label_logprobs) == len(valid_labels):
            # All labels present — confidence is accurate
            result = confidence_from_label_logprobs(label_logprobs)
            return {
                attribute: float(result["label"]),
                'confidence': result["confidence"],
            }
        elif label_logprobs and not all_labels_required:
            # Missing some labels but this is the retry — mark invalid
            return {attribute: "invalid", 'confidence': 0.0}
        elif label_logprobs:
            # Missing some labels — signal retry needed
            return None
        else:
            # No valid labels at all
            generated_text = output.outputs[0].text.strip()
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", generated_text)
            if numbers:
                return {attribute: "invalid", 'confidence': 0.0}
            else:
                return {attribute: "invalid", 'confidence': 0.0}

    def analyze_attribute(self, text, attribute, comment_id=None, annotator_id=None):
        """Analyze a single comment for a single attribute."""
        full_prompt = self.build_prompt(text, attribute, comment_id, annotator_id)

        try:
            # First attempt with logprobs=10
            outputs = self.model.generate([full_prompt], self.sampling_params)
            result = self._extract_result(outputs[0], attribute, all_labels_required=True)

            if result is not None:
                return result

            # Retry with logprobs=20
            outputs = self.model.generate([full_prompt], self.retry_sampling_params)
            result = self._extract_result(outputs[0], attribute, all_labels_required=False)
            return result

        except Exception as e:
            print(f"LLM Error in analyze_attribute({attribute}): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {attribute: "invalid", 'confidence': 0.0}

    def batch_analyze(self, texts, attribute, comment_ids=None, annotator_ids=None):
        """Batch analyze multiple comments for a single attribute."""
        prompts = []
        for i, text in enumerate(texts):
            cid = comment_ids[i] if comment_ids else None
            aid = annotator_ids[i] if annotator_ids else None
            prompt = self.build_prompt(text, attribute, cid, aid)
            prompts.append(prompt)

        # First pass with logprobs=10
        outputs = self.model.generate(prompts, self.sampling_params)

        results = [None] * len(outputs)
        retry_indices = []
        retry_prompts = []

        for i, output in enumerate(outputs):
            result = self._extract_result(output, attribute, all_labels_required=True)
            if result is not None:
                results[i] = result
            else:
                retry_indices.append(i)
                retry_prompts.append(prompts[i])

        # Retry missing ones with logprobs=20
        if retry_prompts:
            retry_outputs = self.model.generate(retry_prompts, self.retry_sampling_params)
            for j, output in enumerate(retry_outputs):
                idx = retry_indices[j]
                results[idx] = self._extract_result(output, attribute, all_labels_required=False)

        return results
