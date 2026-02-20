import re
import math
import numpy as np
import os
import json

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
        full_prompt = f"""<|im_start|>system
You are an expert content moderator.<|im_end|>
<|im_start|>user
    {prompt_text}<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self.model(
                full_prompt,
                max_tokens=10,
                temperature=0.1,
                stop=["<|im_end|>"],
                logprobs=1  # Request logprobs for the top token
            )
            choice = response['choices'][0]
            output = choice['text'].strip()
            
            # Extract number
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
            if numbers:
                val = float(numbers[0])
                
                # Calculate Logit-Based Confidence
                # We use the geometric mean of the probabilities (exp of mean log-probs) of the response tokens
                confidence = 0.0
                if 'logprobs' in choice and choice['logprobs'] and 'token_logprobs' in choice['logprobs']:
                    logprobs = choice['logprobs']['token_logprobs']
                    # Filter out None values just in case
                    valid_logprobs = [lp for lp in logprobs if lp is not None]
                    
                    if valid_logprobs:
                        avg_logprob = np.mean(valid_logprobs)
                        confidence = math.exp(avg_logprob)
                
                # Cap confidence at 1.0 (though exp(logprob) <= 1 always)
                return {attribute: val, 'confidence': confidence}
            else:
                return {attribute: None, 'confidence': 0.0}

        except Exception as e:
            print(f"Error: {e}")
            return {attribute: None, 'confidence': 0.0}
