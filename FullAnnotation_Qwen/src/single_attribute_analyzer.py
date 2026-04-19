import re
import math
import os
import json
from vllm import SamplingParams
from annotator_features_loader import AnnotatorFeaturesLoader


class SingleAttributeAnalyzer:
    def __init__(self, llm_model, use_dynamic=False):
        self.model = llm_model
        self.use_dynamic = use_dynamic
        self.prompts = self._load_prompts()
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
            max_tokens=10,
            temperature=0.1,
            logprobs=5,
            seed=42,
        )

    def _load_prompts(self):
        prompts = {}
        prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')

        if not os.path.exists(prompt_dir):
            raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

        for filename in os.listdir(prompt_dir):
            if filename.endswith(".json") and not filename.startswith("dynamic"):
                attribute = os.path.splitext(filename)[0]
                with open(os.path.join(prompt_dir, filename), 'r') as f:
                    data = json.load(f)
                    if isinstance(data['prompt'], list):
                        prompts[attribute] = '\n'.join(data['prompt'])
                    else:
                        prompts[attribute] = data['prompt']
        return prompts

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

        # Use the model's built-in chat template
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return full_prompt

    def analyze_attribute(self, text, attribute, comment_id=None, annotator_id=None):
        """Analyze a single comment for a single attribute."""
        full_prompt = self.build_prompt(text, attribute, comment_id, annotator_id)

        try:
            outputs = self.model.generate([full_prompt], self.sampling_params)
            output = outputs[0]
            generated_text = output.outputs[0].text.strip()

            # Extract confidence from logprobs
            confidence = 0.0
            if output.outputs[0].logprobs:
                first_token_logprobs = output.outputs[0].logprobs[0]
                if first_token_logprobs:
                    # Get the logprob of the actually sampled token
                    top_logprob = max(first_token_logprobs.values(),
                                      key=lambda x: x.logprob)
                    confidence = math.exp(top_logprob.logprob)

            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", generated_text)
            if numbers:
                return {attribute: float(numbers[0]), 'confidence': confidence}
            else:
                print(f"  No number in response: {repr(generated_text)}")
                return {attribute: None, 'confidence': confidence, 'raw_response': generated_text}

        except Exception as e:
            print(f"LLM Error in analyze_attribute({attribute}): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {attribute: None, 'confidence': 0.0}

    def batch_analyze(self, texts, attribute, comment_ids=None, annotator_ids=None):
        """Batch analyze multiple comments for a single attribute.

        This is significantly faster than calling analyze_attribute one at a time
        because vLLM processes all prompts in a single forward pass with
        continuous batching.
        """
        prompts = []
        for i, text in enumerate(texts):
            cid = comment_ids[i] if comment_ids else None
            aid = annotator_ids[i] if annotator_ids else None
            prompt = self.build_prompt(text, attribute, cid, aid)
            prompts.append(prompt)

        outputs = self.model.generate(prompts, self.sampling_params)

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()

            confidence = 0.0
            if output.outputs[0].logprobs:
                first_token_logprobs = output.outputs[0].logprobs[0]
                if first_token_logprobs:
                    top_logprob = max(first_token_logprobs.values(),
                                      key=lambda x: x.logprob)
                    confidence = math.exp(top_logprob.logprob)

            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", generated_text)
            if numbers:
                results.append({attribute: float(numbers[0]), 'confidence': confidence})
            else:
                results.append({attribute: None, 'confidence': confidence,
                                'raw_response': generated_text})

        return results
