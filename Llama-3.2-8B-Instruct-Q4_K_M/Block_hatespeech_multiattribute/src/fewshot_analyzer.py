from feature_analyzer import FeatureAnalyzer
import re
import math
import numpy as np

class FewShotAnalyzer(FeatureAnalyzer):
    def __init__(self, llm_model):
        super().__init__(llm_model)
        
        # Override the prompt template to include examples
        self.prompt_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert content moderator analyzing online comments. I will provide examples of how to rate comments given SCORING CRITERIA.<|eot_id|><|start_header_id|>user<|end_header_id|>

SCORING CRITERIA (Answer based on the definitions below):
- SENTIMENT (1-5): Question: How would you describe the sentiment? Meaning: Overall emotional tone. 1=Strongly negative, 2=Somewhat negative, 3=Neutral, 4=Somewhat positive, 5=Strongly positive
- RESPECT (1-2): Question: Is this comment respectful towards the group? Meaning: Shows respect/disrespect. 1=Disrespectful/Rude, 2=Respectful/Polite
- INSULT (1-4): Question: Is this comment insulting towards the group? Meaning: Contains insulting language. 1=Strongly Disagree (None), 2=Disagree (Mild), 3=Agree (Clear), 4=Strongly Agree (Severe)
- HUMILIATE (1-3): Question: Is this comment humiliating towards the group? Meaning: Humiliates the group. 1=Strongly Disagree (None), 2=Agree (Attempt), 3=Strongly Agree (Degrading)
- STATUS (1-2): Question: Does this comment frame the group as inferior? Meaning: Framed as inferior or superior. 1=Inferior, 2=Equal/Superior
- DEHUMANIZE (1-2): Question: Does the comment portray the group as less than human? Meaning: Animalistic/vermin comparisons. 1=No, 2=Yes
- VIOLENCE (1-2): Question: Does the comment call for violence against the group? Meaning: Calls for physical harm/death. 1=No, 2=Yes
- GENOCIDE (1-2): Question: Does the comment call for the deliberate killing of a large group? Meaning: Calls for/supports genocide. 1=No, 2=Yes
- ATTACK_DEFEND (1-4): Question: Is the comment attacking or defending the group? Meaning: Explicit aggression or defense. 1=Strongly defending, 2=Defending, 3=Attacking, 4=Strongly attacking
- HATESPEECH (0-1): Question: Does this comment contain hate speech? Meaning: Bias-motivated, hostile, and malicious language targeted at a group. 0=No, 1=Yes
- CONFIDENCE (0.1-1.0): How confident are you in this analysis?

Examples:

Comment: "I hope you all die, you filthy animals."
SENTIMENT 1
RESPECT 1
INSULT 4
HUMILIATE 3
STATUS 1
DEHUMANIZE 2
VIOLENCE 2
GENOCIDE 1
ATTACK_DEFEND 4
HATESPEECH 1
CONFIDENCE 1.0

Comment: "I disagree with your opinion, but I respect your right to say it."
SENTIMENT 3
RESPECT 2
INSULT 1
HUMILIATE 1
STATUS 2
DEHUMANIZE 1
VIOLENCE 1
GENOCIDE 1
ATTACK_DEFEND 2
HATESPEECH 0
CONFIDENCE 0.9

Comment: "{text}"

RESPOND WITH ONLY NUMBERS IN THE SAME FORMAT.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''

    def analyze(self, text):
        """Analyze using few-shot prompt template"""
        try:
            prompt = self.prompt_template.format(text=text[:500])

            response = self.model(
                prompt,
                max_tokens=150,
                temperature=0.1,
                stop=["<|eot_id|>"],
                logprobs=1
            )

            output = response['choices'][0]['text'].strip().lower()

            # Parse the output using parent method
            results = self._parse_output(output)
            results['raw_output'] = output[:200]
            results['text'] = text[:100]
            results['analysis_type'] = 'fewshot'

            # Calculate Logit-Based Confidence
            confidence = 0.0
            choice = response['choices'][0]
            if 'logprobs' in choice and choice['logprobs'] and 'token_logprobs' in choice['logprobs']:
                logprobs = choice['logprobs']['token_logprobs']
                # Filter out None values
                valid_logprobs = [lp for lp in logprobs if lp is not None]
                
                if valid_logprobs:
                    avg_logprob = np.mean(valid_logprobs)
                    confidence = math.exp(avg_logprob)
            
            # Store logit confidence
            results['confidence'] = confidence

            return results

            return results

        except Exception as e:
            print(f"❌ Error analyzing (fewshot): {e}")
            return self._fallback_analysis(text)
