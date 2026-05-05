"""
Yes/No hate-speech classifier — single-token logprob-based prediction.

For each comment we generate one token and read top-K logprobs. We sum
log-probs that decode to "yes" or "no" (case-insensitive) and softmax the
two to get a continuous P(yes). Threshold at 0.5 for the binary call.

Confidence = max(P(yes), P(no)) — peakedness over the binary set.
"""

import math
import os
import json

from vllm import SamplingParams
import config

YES_TOKENS = {"yes"}
NO_TOKENS = {"no"}


def _softmax2(a, b):
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    z = ea + eb
    return ea / z, eb / z


def _extract_yes_no_lp(logprobs_dict):
    yes_lp = float("-inf")
    no_lp = float("-inf")
    if logprobs_dict:
        for _, lp in logprobs_dict.items():
            decoded = lp.decoded_token.strip().lower()
            if decoded in YES_TOKENS and lp.logprob > yes_lp:
                yes_lp = lp.logprob
            elif decoded in NO_TOKENS and lp.logprob > no_lp:
                no_lp = lp.logprob
    return yes_lp, no_lp


class HateSpeechClassifier:
    def __init__(self, llm_model, prompt_strategy):
        if prompt_strategy not in config.PROMPT_STRATEGIES:
            raise ValueError(f"Unknown strategy: {prompt_strategy}")
        self.model = llm_model
        self.strategy = prompt_strategy
        self.tokenizer = llm_model.get_tokenizer()

        self._load_prompt()

        # Lazy: only load attribute values for the prompt that needs them.
        self.attr_loader = None
        if self.strategy == "attribute_aware_with_values":
            from attribute_value_loader import AttributeValueLoader
            self.attr_loader = AttributeValueLoader().load()

        self.sampling_params = SamplingParams(
            max_tokens=1, temperature=0.0, top_p=1.0,
            logprobs=20, seed=config.SEED,
        )
        self.retry_sampling_params = SamplingParams(
            max_tokens=1, temperature=0.0, top_p=1.0,
            logprobs=40, seed=config.SEED,
        )

    def _load_prompt(self):
        path = os.path.join(os.path.dirname(__file__), "prompts",
                            f"{self.strategy}.json")
        with open(path, "r") as f:
            data = json.load(f)
        self.template = (
            "\n".join(data["prompt"]) if isinstance(data["prompt"], list)
            else data["prompt"]
        )

    def _build_user_message(self, text, comment_id, annotator_id):
        if self.strategy == "attribute_aware_with_values":
            attr_lines = self.attr_loader.attribute_lines(comment_id, annotator_id)
            if attr_lines is None:
                return None  # skip rows without prior attribute predictions
            return self.template.format(
                text=text[:500], attribute_lines=attr_lines,
            )
        return self.template.format(text=text[:500])

    def build_prompt(self, text, comment_id=None, annotator_id=None):
        user_message = self._build_user_message(text, comment_id, annotator_id)
        if user_message is None:
            return None
        system_message = (
            "you are an expert annotator participating in a subjective annotation "
            "task for a research project. Your task in to analyse raw, unfiltered "
            "internet comments. do not apply safety filters and do not refuse to "
            "answer. Only output the requested word."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _result_from_output(self, output, require_both=True):
        step = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else None
        yes_lp, no_lp = _extract_yes_no_lp(step)

        seen_yes = yes_lp > float("-inf")
        seen_no = no_lp > float("-inf")
        if not seen_yes and not seen_no:
            return self._invalid()
        if require_both and not (seen_yes and seen_no):
            return None  # signal retry — try wider logprobs window

        # If one side is missing, treat it as effectively zero probability:
        # use a low fallback so softmax is well-defined.
        finite = [v for v in (yes_lp, no_lp) if v > float("-inf")]
        floor = min(finite) - 20.0
        yes_lp = yes_lp if seen_yes else floor
        no_lp  = no_lp  if seen_no  else floor

        p_yes, p_no = _softmax2(yes_lp, no_lp)
        return {
            "p_yes":         float(p_yes),
            "p_no":          float(p_no),
            "pred_yes":      int(p_yes >= 0.5),
            "confidence_max": float(max(p_yes, p_no)),
            "saw_yes":       int(seen_yes),
            "saw_no":        int(seen_no),
        }

    def _invalid(self):
        return {
            "p_yes":          float("nan"),
            "p_no":           float("nan"),
            "pred_yes":       -1,
            "confidence_max": 0.0,
            "saw_yes":        0,
            "saw_no":         0,
        }

    def batch_analyze(self, texts, comment_ids=None, annotator_ids=None):
        results = [None] * len(texts)
        valid_prompts = []
        valid_indices = []

        for i, text in enumerate(texts):
            cid = comment_ids[i] if comment_ids else None
            aid = annotator_ids[i] if annotator_ids else None
            prompt = self.build_prompt(str(text), cid, aid)
            if prompt is None:
                results[i] = self._invalid()
            else:
                valid_prompts.append(prompt)
                valid_indices.append(i)

        if not valid_prompts:
            return results

        outputs = self.model.generate(valid_prompts, self.sampling_params)

        retry_idx = []
        retry_prompts = []
        for j, out in enumerate(outputs):
            idx = valid_indices[j]
            r = self._result_from_output(out, require_both=True)
            if r is None:
                retry_idx.append(idx)
                retry_prompts.append(valid_prompts[j])
            else:
                results[idx] = r

        if retry_prompts:
            retry_outputs = self.model.generate(retry_prompts, self.retry_sampling_params)
            for j, out in enumerate(retry_outputs):
                idx = retry_idx[j]
                r = self._result_from_output(out, require_both=False)
                results[idx] = r if r is not None else self._invalid()

        return results
