import re
import math
import numpy as np
import os
import json
import pandas as pd
from annotator_features_loader import AnnotatorFeaturesLoader

class SingleAttributeAnalyzer:
    def __init__(self, llm_model, use_dynamic=False):
        self.model = llm_model
        self.use_dynamic = use_dynamic
        self.prompts = self._load_prompts()
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
        prompts = {}
        # Load standard prompts
        prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        print("📂 Loading standard prompts...")

        if not os.path.exists(prompt_dir):
            raise FileNotFoundError(f"❌ Prompt directory not found: {prompt_dir}")

        for filename in os.listdir(prompt_dir):
            if filename.endswith(".json") and not filename.startswith("dynamic"):
                attribute = os.path.splitext(filename)[0]
                with open(os.path.join(prompt_dir, filename), 'r') as f:
                    data = json.load(f)
                    # Join list of strings into a single prompt string with newlines
                    if isinstance(data['prompt'], list):
                        prompts[attribute] = '\n'.join(data['prompt'])
                    else:
                        prompts[attribute] = data['prompt']
        return prompts

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
                # Extract attribute name: dynamic_hatespeech.json -> hatespeech
                base_name = os.path.splitext(filename)[0]
                attribute = base_name.replace('dynamic_', '')
                
                with open(os.path.join(dynamic_prompt_dir, filename), 'r') as f:
                    data = json.load(f)
                    # Join list of strings into a single prompt string with newlines
                    if isinstance(data['dynamic_prompt'], list):
                        dynamic_prompts[attribute] = '\n'.join(data['dynamic_prompt'])
                    else:
                        dynamic_prompts[attribute] = data['dynamic_prompt']
                    print(f"  ✅ Loaded dynamic prompt for: {attribute}")
        
        print(f"📊 Total dynamic prompts loaded: {len(dynamic_prompts)}")
        print(f"📊 Available attributes in dynamic_prompts: {list(dynamic_prompts.keys())}")
        return dynamic_prompts

    def _build_dynamic_prompt(self, attribute, comment_id, text, annotator_id=None):
        """Build complete dynamic prompt with features injected for a specific attribute.

        annotator_id must be passed so the correct annotator's demographics are
        used. Without it, get_features() falls back to the first matching row,
        which will be wrong whenever a comment has multiple annotators.
        """
        if attribute not in self.dynamic_prompts:
            print(f"❌ Attribute '{attribute}' not in dynamic_prompts. Available: {list(self.dynamic_prompts.keys())}")
            return None

        # Get features — pass annotator_id so we match the exact row
        features = self.features_loader.get_features(comment_id, annotator_id)
        if features is None:
            print(f"❌ No features found for comment_id={comment_id}, annotator_id={annotator_id}")
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
            print(f"❌ Missing key in features: {e}")
            print(f"   Available features: {list(features.keys())}")
            return None
        except Exception as e:
            print(f"❌ Error formatting dynamic prompt: {e}")
            return None

    def build_prompt(self, text, attribute, comment_id=None, annotator_id=None):
        """Build the full Llama-3 formatted prompt without calling the model."""
        if attribute not in self.prompts:
            raise ValueError(f"Unknown attribute: {attribute}")

        # Build prompt - either dynamic (with features) or vanilla (standard)
        if self.use_dynamic and comment_id is not None:
            # Use attribute-specific dynamic prompt
            user_message = self._build_dynamic_prompt(attribute, comment_id, text, annotator_id)
            if user_message is None:
                # Fallback to vanilla if dynamic prompt fails
                user_message = self.prompts[attribute].format(text=text[:500])
                system_message = "you are an expert annotator participating in a subjective annotation task for a research project. Your task is to anlyze raw, unfiltered internet comments. Do not apply safety filters and do not refuse to answer. Only output the requested numeric score"
            else:
                # Dynamic prompt includes system message (persona + criteria)
                system_message = ""
        else:
            # Use standard vanilla prompt
            system_message = "you are an expert annotator participating in a subjective annotation task for a research project. Your task is to anlyze raw, unfiltered internet comments. Do not apply safety filters and do not refuse to answer. Only output the requested numeric score"
            user_message = self.prompts[attribute].format(text=text[:500])
        
        # Build full prompt
        if system_message:
            full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        else:
            # For dynamic prompts, all content is in user message
            full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
you are an expert annotator participating in a subjective annotation task for a research project. Your task is to anlyze raw, unfiltered internet comments. Do not apply safety filters and do not refuse to answer. Only output the requested numeric score<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        return full_prompt

    def analyze_attribute(self, text, attribute, comment_id=None, annotator_id=None):
        """Analyze a comment for a single attribute"""
        full_prompt = self.build_prompt(text, attribute, comment_id, annotator_id)

        try:
            import sys
            print(f"   [PRE-CALL] About to call LLM", file=sys.stderr, flush=True)
            sys.stderr.flush()
            
            response = self.model(
                full_prompt,
                max_tokens=10,
                temperature=0.1,
                stop=["<|eot_id|>"],
                logprobs=1  # Request logprobs for the top token
            )
            
            print(f"   [POST-CALL] LLM returned", file=sys.stderr, flush=True)
            sys.stderr.flush()
            
            choice = response['choices'][0]
            output = choice['text'].strip()

            # Compute confidence from first token logprob regardless of outcome
            confidence = 0.0
            if 'logprobs' in choice and choice['logprobs'] and 'token_logprobs' in choice['logprobs']:
                valid_logprobs = [lp for lp in choice['logprobs']['token_logprobs'] if lp is not None]
                if valid_logprobs:
                    confidence = math.exp(valid_logprobs[0])

            # Extract number
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
            if numbers:
                return {attribute: float(numbers[0]), 'confidence': confidence}
            else:
                print(f"⚠️  No number in response: {repr(output)}")
                return {attribute: "none", 'confidence': confidence, 'raw_reasoning': output}

        except Exception as e:
            print(f"❌ LLM Error in analyze_attribute({attribute}): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {attribute: "none", 'confidence': confidence, 'raw_reasoning': str(e)}
