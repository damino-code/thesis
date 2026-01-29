import re
import time
import math
import numpy as np
from datetime import datetime

class FeatureAnalyzer:
    def __init__(self, llm_model):
        self.model = llm_model

        # Define demographic profiles
        self.demographic_profiles = {
            "black_muslim_woman": {
                "description": "you are a black Muslim woman",
                "details": "You identify as a Black Muslim woman with lived experience of intersectional discrimination."
            },
            "white_christian_man": {
                "description": "you are a white Christian man",
                "details": "You identify as a white Christian man from a traditional background."
            },
            "latina_atheist_woman": {
                "description": "you are a Latina atheist woman",
                "details": "You identify as a Latina atheist woman with progressive views."
            },
            "asian_buddhist_man": {
                "description": "you are an Asian Buddhist man",
                "details": "You identify as an Asian Buddhist man with moderate political views."
            },
            "mixed_race_agnostic_nonbinary": {
                "description": "you are a mixed-race agnostic non-binary person",
                "details": "You identify as a mixed-race agnostic non-binary person with diverse perspectives."
            },
            "neutral_baseline": {
                "description": "you are an objective annotator",
                "details": "You are an objective annotator without specified demographic characteristics."
            }
        }

    def create_profile_prompt(self, text, profile_name):
        """Create a prompt with demographic perspective"""
        if profile_name not in self.demographic_profiles:
            raise ValueError(f"Unknown profile: {profile_name}")

        profile = self.demographic_profiles[profile_name]

        prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a content moderator. {profile['description']}.
{profile['details']}<|eot_id|><|start_header_id|>user<|end_header_id|>

As someone with your background, analyze this comment for multiple attributes from YOUR perspective:

Comment: "{text}"

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

RESPOND WITH ONLY NUMBERS IN THIS FORMAT:
SENTIMENT [number]
RESPECT [number]
INSULT [number]
HUMILIATE [number]
STATUS [number]
DEHUMANIZE [number]
VIOLENCE [number]
GENOCIDE [number]
ATTACK_DEFEND [number]
HATESPEECH [number]
CONFIDENCE [number]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
        return prompt

    def analyze(self, text, profile_name="neutral_baseline"):
        """Analyze text with a specific demographic profile (defaults to neutral)"""
        return self.analyze_with_profile(text, profile_name)

    def analyze_with_profile(self, text, profile_name):
        """Analyze text with a specific demographic profile"""
        try:
            prompt = self.create_profile_prompt(text[:500], profile_name)

            response = self.model(
                prompt,
                max_tokens=150,
                temperature=0.1,
                stop=["<|eot_id|>"],
                logprobs=1
            )

            output = response['choices'][0]['text'].strip().lower()

            # Parse the output
            results = self._parse_output(output)
            results['profile'] = profile_name
            results['raw_output'] = output[:200]
            results['text'] = text[:100]

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
            
            # Store logit confidence (overwriting parsed confidence if desired, or as separate field)
            # User asked to print it as 'confidence' later, so we overwrite.
            results['confidence'] = confidence

            return results

        except Exception as e:
            print(f"❌ Error analyzing with {profile_name}: {e}")
            return self._fallback_analysis(text, profile_name)

    def _parse_output(self, output):
        """Parse LLM output to extract numeric values"""
        results = {
            'sentiment': 3.0,
            'respect': 1.5,
            'insult': 1.0,
            'humiliate': 1.0,
            'status': 1.5,
            'dehumanize': 1.0,
            'violence': 1.0,
            'genocide': 1.0,
            'attack_defend': 2.0,
            'hatespeech': 0.0,
            'confidence': 0.7
        }

        # Extract numbers in sequence
        number_sequence = re.findall(r'([0-9]\.?[0-9]*)', output)

        if len(number_sequence) >= 10:
            try:
                sentiment = float(number_sequence[0])
                respect = float(number_sequence[1])
                insult = float(number_sequence[2])
                humiliate = float(number_sequence[3])
                status = float(number_sequence[4])
                dehumanize = float(number_sequence[5])
                violence = float(number_sequence[6])
                genocide = float(number_sequence[7])
                attack_defend = float(number_sequence[8])
                hatespeech = float(number_sequence[9])
                confidence = float(number_sequence[10]) if len(number_sequence) > 10 else 0.7

                # Validate and assign
                if 1.0 <= sentiment <= 5.0:
                    results['sentiment'] = sentiment
                if 1.0 <= respect <= 2.0:
                    results['respect'] = respect
                if 1.0 <= insult <= 4.0:
                    results['insult'] = insult
                if 1.0 <= humiliate <= 3.0:
                    results['humiliate'] = humiliate
                if 1.0 <= status <= 2.0:
                    results['status'] = status
                if 1.0 <= dehumanize <= 2.0:
                    results['dehumanize'] = dehumanize
                if 1.0 <= violence <= 2.0:
                    results['violence'] = violence
                if 1.0 <= genocide <= 2.0:
                    results['genocide'] = genocide
                if 1.0 <= attack_defend <= 4.0:
                    results['attack_defend'] = attack_defend
                if 0.0 <= hatespeech <= 1.0:
                    results['hatespeech'] = hatespeech
                if 0.1 <= confidence <= 1.0:
                    results['confidence'] = confidence

            except (ValueError, IndexError):
                pass

        return results

    def _fallback_analysis(self, text, profile_name="unknown"):
        """Fallback analysis using keyword detection"""
        text_lower = text.lower()

        results = {
            'sentiment': 3.0,
            'respect': 2.0,
            'insult': 1.0,
            'humiliate': 1.0,
            'status': 2.0,
            'dehumanize': 1.0,
            'violence': 1.0,
            'genocide': 1.0,
            'attack_defend': 2.0,
            'hatespeech': 0.0,
            'confidence': 0.5,
            'profile': profile_name,
            'raw_output': 'fallback_analysis',
            'text': text[:100]
        }
        
        # Simple fallback checks
        hate_words = ['hate', 'kill', 'die', 'murder', 'eliminate', 'destroy']
        slurs = ['slur', 'racist', 'sexist']

        if any(word in text_lower for word in hate_words):
            results['sentiment'] = 1.0
            results['respect'] = 1.0
            results['insult'] = 4.0
            results['attack_defend'] = 4.0
            results['hatespeech'] = 1.0
            
        if any(word in text_lower for word in slurs):
            results['hatespeech'] = 1.0
            results['insult'] = 4.0
            
        return results
