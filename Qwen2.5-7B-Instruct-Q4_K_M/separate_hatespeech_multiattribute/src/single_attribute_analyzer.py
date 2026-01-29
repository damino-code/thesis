import re
import math
import numpy as np

class SingleAttributeAnalyzer:
    def __init__(self, llm_model):
        self.model = llm_model
        
        self.prompts = {
            'sentiment': """
SCORING CRITERIA (Answer based on the definitions below):
- SENTIMENT (1-5): Question: How would you describe the sentiment? Meaning: Overall emotional tone. 1=Strongly negative, 2=Somewhat negative, 3=Neutral, 4=Somewhat positive, 5=Strongly positive

Analyze this comment for SENTIMENT:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (1-5).
""",
            'respect': """
SCORING CRITERIA (Answer based on the definitions below):
- RESPECT (1-2): Question: Is this comment respectful towards the group? Meaning: Shows respect/disrespect. 1=Disrespectful/Rude, 2=Respectful/Polite

Analyze this comment for RESPECT:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (1-2).
""",
            'insult': """
SCORING CRITERIA (Answer based on the definitions below):
- INSULT (1-4): Question: Is this comment insulting towards the group? Meaning: Contains insulting language. 1=Strongly Disagree (None), 2=Disagree (Mild), 3=Agree (Clear), 4=Strongly Agree (Severe)

Analyze this comment for INSULT:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (1-4).
""",
            'humiliate': """
SCORING CRITERIA (Answer based on the definitions below):
- HUMILIATE (1-3): Question: Is this comment humiliating towards the group? Meaning: Humiliates the group. 1=Strongly Disagree (None), 2=Agree (Attempt), 3=Strongly Agree (Degrading)

Analyze this comment for HUMILIATION:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (1-3).
""",
            'status': """
SCORING CRITERIA (Answer based on the definitions below):
- STATUS (1-2): Question: Does this comment frame the group as inferior? Meaning: Framed as inferior or superior. 1=Inferior, 2=Equal/Superior

Analyze this comment for STATUS:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (1-2).
""",
            'dehumanize': """
SCORING CRITERIA (Answer based on the definitions below):
- DEHUMANIZE (1-2): Question: Does the comment portray the group as less than human? Meaning: Animalistic/vermin comparisons. 1=No, 2=Yes

Analyze this comment for DEHUMANIZATION:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (1-2).
""",
            'violence': """
SCORING CRITERIA (Answer based on the definitions below):
- VIOLENCE (1-2): Question: Does the comment call for violence against the group? Meaning: Calls for physical harm/death. 1=No, 2=Yes

Analyze this comment for VIOLENCE:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (1-2).
""",
            'genocide': """
SCORING CRITERIA (Answer based on the definitions below):
- GENOCIDE (1-2): Question: Does the comment call for the deliberate killing of a large group? Meaning: Calls for/supports genocide. 1=No, 2=Yes

Analyze this comment for GENOCIDE:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (1-2).
""",
            'attack_defend': """
SCORING CRITERIA (Answer based on the definitions below):
- ATTACK_DEFEND (1-4): Question: Is the comment attacking or defending the group? Meaning: Explicit aggression or defense. 1=Strongly defending, 2=Defending, 3=Attacking, 4=Strongly attacking

Analyze this comment for ATTACK vs DEFENSE:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (1-4).
""",
            'hatespeech': """
SCORING CRITERIA (Answer based on the definitions below):
- HATESPEECH (0-1): Question: Does this comment contain hate speech? Meaning: Bias-motivated, hostile, and malicious language targeted at a group. 0=No, 1=Yes

Analyze this comment for HATE SPEECH:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0 or 1).
"""
        }

    def analyze_attribute(self, text, attribute):
        """Analyze a comment for a single attribute"""
        if attribute not in self.prompts:
            raise ValueError(f"Unknown attribute: {attribute}")
            
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert content moderator.<|eot_id|><|start_header_id|>user<|end_header_id|>
{self.prompts[attribute].format(text=text[:500])}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        try:
            response = self.model(
                full_prompt,
                max_tokens=10,
                temperature=0.1,
                stop=["<|eot_id|>"],
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
