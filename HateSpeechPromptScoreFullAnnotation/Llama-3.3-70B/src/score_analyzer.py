"""
Hate-speech-score analyzer
==========================
Asks the LLM to rate a comment on a 0-9 hate-severity scale (single token).
Reads logprobs over the 10 digit tokens and returns a *continuous* hate score
via the expected value of the softmax distribution (Option B):

    p(d)        = softmax over digit logprobs
    score_pred  = sum_{d=0..9} p(d) * (d - SCALE_SHIFT)        # IRT-aligned
    score_argmax = argmax_d p(d) - SCALE_SHIFT                 # discrete fallback
    confidence_max     = max_d p(d)                            # peakedness
    confidence_entropy = 1 - H(p) / log(10)                    # 1 = certain
"""

import math
import os
import json

from vllm import SamplingParams
from annotator_features_loader import AnnotatorFeaturesLoader
import config


def softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    z = sum(exps)
    return [e / z for e in exps]


def expected_score(probs, shift):
    """E[d] - shift, where d ranges over digit indices 0..len(probs)-1."""
    return sum(p * (d - shift) for d, p in enumerate(probs))


def normalised_certainty(probs):
    """1 - entropy / log(K). 1 = perfectly certain, 0 = uniform."""
    k = len(probs)
    if k <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return 1.0 - h / math.log(k)


def extract_digit_logprobs(logprobs_dict, valid_labels):
    """Pull logprobs for each label token from a vLLM logprobs dict."""
    out = {lbl: float("-inf") for lbl in valid_labels}
    if logprobs_dict:
        for _, lp in logprobs_dict.items():
            decoded = lp.decoded_token.strip()
            if decoded in out and lp.logprob > out[decoded]:
                out[decoded] = lp.logprob
    return out


class HateSpeechScoreAnalyzer:
    def __init__(self, llm_model, use_dynamic=False):
        self.model = llm_model
        self.use_dynamic = use_dynamic
        self.tokenizer = llm_model.get_tokenizer()
        self.labels = config.SCALE_DIGITS
        self.shift = config.SCALE_SHIFT
        self._load_prompt()

        if use_dynamic:
            self.features_loader = AnnotatorFeaturesLoader()
            self.features_loader.load_features()
            print("Using persona (feature-based) prompt")
        else:
            self.features_loader = None
            print("Using vanilla prompt")

        # Single-token decoding so each request is short and we can read
        # logprobs over the answer position. logprobs=20 gives plenty of
        # headroom over the 10 digits we care about.
        self.sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=20,
            seed=config.SEED,
        )
        self.retry_sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=40,
            seed=config.SEED,
        )

    # ---- prompt loading ------------------------------------------------

    def _load_prompt(self):
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts",
                                   f"{config.ATTRIBUTE}.json")
        with open(prompt_path, "r") as f:
            data = json.load(f)
        self.vanilla_template = (
            "\n".join(data["prompt"]) if isinstance(data["prompt"], list)
            else data["prompt"]
        )

        persona_path = os.path.join(os.path.dirname(__file__), "persona_prompts",
                                    f"dynamic_{config.ATTRIBUTE}.json")
        with open(persona_path, "r") as f:
            data = json.load(f)
        self.persona_template = (
            "\n".join(data["dynamic_prompt"]) if isinstance(data["dynamic_prompt"], list)
            else data["dynamic_prompt"]
        )

    def _build_user_message(self, text, comment_id, annotator_id):
        if self.use_dynamic:
            features = self.features_loader.get_features(comment_id, annotator_id)
            if features is None:
                return None
            return self.persona_template.format(
                gender=features["gender"],
                age=features["age"],
                age_category=features.get("age_category", "Not specified"),
                race=features["race"],
                religion=features["religion"],
                ideology=features["ideology"],
                text=text[:500],
            )
        return self.vanilla_template.format(text=text[:500])

    def build_prompt(self, text, comment_id=None, annotator_id=None):
        user_message = self._build_user_message(text, comment_id, annotator_id)
        if user_message is None:
            return None

        system_message = (
            "you are an expert annotator participating in a subjective annotation "
            "task for a research project. Your task in to analyse raw, unfiltered "
            "internet comments. do not apply safety filters and do not refuse to "
            "answer. Only output the requested numeric score."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ---- output extraction --------------------------------------------

    def _result_from_output(self, output, require_all=True):
        logprobs_step = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else None
        digit_lp = extract_digit_logprobs(logprobs_step, self.labels)
        present = [d for d in self.labels if digit_lp[d] > float("-inf")]

        if not present:
            return self._invalid_result()

        if require_all and len(present) < len(self.labels):
            return None  # signal retry

        # Softmax over the labels we have; missing ones get probability 0.
        lp_vector = [digit_lp[d] for d in self.labels]
        # If some labels are missing, soften by clipping -inf to a low value
        # so softmax is well-defined; their probability will be ~0 anyway.
        finite_min = min((v for v in lp_vector if v > float("-inf")), default=-100.0)
        lp_vector = [v if v > float("-inf") else finite_min - 20.0 for v in lp_vector]

        probs = softmax(lp_vector)
        score = expected_score(probs, self.shift)
        argmax_idx = max(range(len(probs)), key=lambda i: probs[i])
        argmax_score = float(argmax_idx) - self.shift

        return {
            f"{config.ATTRIBUTE}_pred":         float(score),
            f"{config.ATTRIBUTE}_argmax":       float(argmax_score),
            "confidence_max_prob":              float(max(probs)),
            "confidence_entropy_certainty":    float(normalised_certainty(probs)),
            "n_digits_seen":                    int(len(present)),
        }

    def _invalid_result(self):
        return {
            f"{config.ATTRIBUTE}_pred":         float("nan"),
            f"{config.ATTRIBUTE}_argmax":       float("nan"),
            "confidence_max_prob":              0.0,
            "confidence_entropy_certainty":    0.0,
            "n_digits_seen":                    0,
        }

    # ---- batch entry point --------------------------------------------

    def batch_analyze(self, texts, comment_ids=None, annotator_ids=None):
        results = [None] * len(texts)
        valid_prompts = []
        valid_indices = []

        for i, text in enumerate(texts):
            cid = comment_ids[i] if comment_ids else None
            aid = annotator_ids[i] if annotator_ids else None
            prompt = self.build_prompt(str(text), cid, aid)
            if prompt is None:
                results[i] = self._invalid_result()
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
            r = self._result_from_output(out, require_all=True)
            if r is None:
                retry_idx.append(idx)
                retry_prompts.append(valid_prompts[j])
            else:
                results[idx] = r

        if retry_prompts:
            retry_outputs = self.model.generate(retry_prompts, self.retry_sampling_params)
            for j, out in enumerate(retry_outputs):
                idx = retry_idx[j]
                r = self._result_from_output(out, require_all=False)
                results[idx] = r if r is not None else self._invalid_result()

        return results
