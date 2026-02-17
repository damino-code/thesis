import re
import math
import numpy as np
import os
import json

import config

class SingleAttributeAnalyzer:
    def __init__(self, llm_model):
        self.model = llm_model
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        prompts = {}
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        for filename in os.listdir(prompt_dir):
            if filename.endswith(".json"):
                attribute = os.path.splitext(filename)[0]
                with open(os.path.join(prompt_dir, filename), "r") as handle:
                    data = json.load(handle)
                    if isinstance(data.get("prompt"), list):
                        prompts[attribute] = "\n".join(data["prompt"])
                    else:
                        prompts[attribute] = data.get("prompt", "")
        return prompts

    def analyze_attribute(self, text, attribute):
        """Analyze a comment for a single attribute"""
        if attribute not in self.prompts:
            raise ValueError(f"Unknown attribute: {attribute}")
            
        prompt_text = self.prompts[attribute].format(text=text[:500])
        messages = [
            {"role": "system", "content": "You are an expert content moderator."},
            {"role": "user", "content": prompt_text},
        ]

        try:
            response = self.model.chat(
                messages=messages,
                max_tokens=10,
                temperature=0.1,
                stop=None,
                logprobs=True,
            )
            choice = response["choices"][0]
            output = choice.get("message", {}).get("content", "").strip()
            
            # Extract number
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
            if numbers:
                val = float(numbers[0])
                
                # Calculate Logit-Based Confidence
                # We use the geometric mean of the probabilities (exp of mean log-probs) of the response tokens
                confidence = config.DEFAULT_CONFIDENCE
                logprobs_payload = choice.get("logprobs")
                if logprobs_payload:
                    token_logprobs = []
                    if isinstance(logprobs_payload, dict):
                        content_logprobs = logprobs_payload.get("content")
                        if isinstance(content_logprobs, list):
                            token_logprobs = [item.get("logprob") for item in content_logprobs]
                        else:
                            token_logprobs = logprobs_payload.get("token_logprobs", [])

                    valid_logprobs = [lp for lp in token_logprobs if lp is not None]
                    if valid_logprobs:
                        avg_logprob = np.mean(valid_logprobs)
                        confidence = math.exp(avg_logprob)
                
                # Cap confidence at 1.0 (though exp(logprob) <= 1 always)
                return {attribute: val, "confidence": confidence}
            else:
                return {attribute: None, "confidence": 0.0}

        except Exception as e:
            print(f"Error: {e}")
            return {attribute: None, "confidence": 0.0}
