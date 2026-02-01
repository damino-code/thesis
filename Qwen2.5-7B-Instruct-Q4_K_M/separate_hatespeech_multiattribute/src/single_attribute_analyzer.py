import re
import math
import numpy as np

class SingleAttributeAnalyzer:
    def __init__(self, llm_model):
        self.model = llm_model
        
        self.prompts = {
            'sentiment': """
SCORING CRITERIA (Answer based on the definitions below):
- SENTIMENT (0-4): Question: How would you describe the sentiment? 0=Strongly negative, 1=Somewhat negative, 2=Neutral, 3=Somewhat positive, 4=Strongly positive

Analyze this comment for SENTIMENT:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
""",
            'respect': """
SCORING CRITERIA (Answer based on the definitions below):
- RESPECT (0-4): Question: Is this comment respectful towards the group? 0=Strongly disrespectful, 1=Disrespectful/Rude, 2=Neutral, 3=Respectful/Polite, 4=Strongly respectful

Analyze this comment for RESPECT:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
""",
            'insult': """
SCORING CRITERIA (Answer based on the definitions below):
- INSULT (0-4): Question: Is this comment insulting towards the group? 0=Strongly Disagree (None), 1=Disagree (Mild), 2=Neutral/Unsure, 3=Agree (Clear), 4=Strongly Agree (Severe)

Analyze this comment for INSULT:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
""",
            'humiliate': """
SCORING CRITERIA (Answer based on the definitions below):
- HUMILIATE (0-4): Question: Is this comment humiliating towards the group? 0=Strongly Disagree (None), 1=Disagree (Mild), 2=Neutral/Unsure, 3=Agree (Attempt), 4=Strongly Agree (Degrading)

Analyze this comment for HUMILIATION:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
""",
            'status': """
SCORING CRITERIA (Answer based on the definitions below):
- STATUS (0-4): Question: Does this comment frame the group as inferior? 0=Strongly inferior, 1=Inferior, 2=Equal/Neutral, 3=Superior, 4=Strongly superior

Analyze this comment for STATUS:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
""",
            'dehumanize': """
SCORING CRITERIA (Answer based on the definitions below):
- DEHUMANIZE (0-4): Question: Does the comment portray the group as less than human? 0=Strongly no, 1=No, 2=Unclear/Neutral, 3=Yes, 4=Strongly yes

Analyze this comment for DEHUMANIZATION:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
""",
            'violence': """
SCORING CRITERIA (Answer based on the definitions below):
- VIOLENCE (0-4): Question: Does the comment call for violence against the group? 0=Strongly no, 1=No, 2=Unclear/Neutral, 3=Yes, 4=Strongly yes

Analyze this comment for VIOLENCE:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
""",
            'genocide': """
SCORING CRITERIA (Answer based on the definitions below):
- GENOCIDE (0-4): Question: Does the comment call for the deliberate killing of a large group? 0=Strongly no, 1=No, 2=Unclear/Neutral, 3=Yes, 4=Strongly yes

Analyze this comment for GENOCIDE:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
""",
            'attack_defend': """
SCORING CRITERIA (Answer based on the definitions below):
- ATTACK_DEFEND (0-4): Question: Is the comment attacking or defending the group? 0=Strongly defending, 1=Defending, 2=Neutral/Mixed, 3=Attacking, 4=Strongly attacking

Analyze this comment for ATTACK vs DEFENSE:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
""",
            'hatespeech': """
SCORING CRITERIA (Answer based on the definitions below):
- HATESPEECH (0-4): Question: Does this comment contain hate speech? 0=Strongly no, 1=No, 2=Unclear/Neutral, 3=Yes, 4=Strongly yes

Analyze this comment for HATE SPEECH:
Comment: "{text}"

RESPOND WITH ONLY THE NUMBER (0-4).
"""
        }

    def analyze_attribute(self, text, attribute):
        """Analyze a comment for a single attribute"""
        if attribute not in self.prompts:
            raise ValueError(f"Unknown attribute: {attribute}")
            
        full_prompt = f"""<|im_start|>system
You are an expert content moderator.<|im_end|>
<|im_start|>user
{self.prompts[attribute].format(text=text[:500])}<|im_end|>
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
