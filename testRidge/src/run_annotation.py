"""
Annotate test_2000.csv with Llama-3.1-70B (vanilla mode).

For each of the 10 attributes, runs all 2000 texts through the prompt and
saves per-attribute results to results/per_attribute/<attribute>/.
Run merge_results.py afterwards to combine into a single CSV.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

import config

# ---------------------------------------------------------------------------
# Add FullAnnotation_LLama/src to path so we can reuse the model loader
# ---------------------------------------------------------------------------
LLAMA_SRC = config.THESIS_DIR / "FullAnnotation_LLama" / "src"
sys.path.insert(0, str(LLAMA_SRC))


# ---------------------------------------------------------------------------
# Helpers (self-contained, no dependency on FullAnnotation internals)
# ---------------------------------------------------------------------------

def softmax_dict(score_dict):
    m = max(score_dict.values())
    exps = {k: math.exp(v - m) for k, v in score_dict.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def confidence_from_logprobs(label_logprobs):
    probs = softmax_dict(label_logprobs)
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    label, conf = ranked[0]
    return label, conf


def load_prompt(attribute):
    path = config.PROMPTS_DIR / f"{attribute}.json"
    with open(path) as f:
        data = json.load(f)
    template = "\n".join(data["prompt"]) if isinstance(data["prompt"], list) else data["prompt"]
    labels = data.get("labels", ["0", "1", "2", "3", "4"])
    return template, labels


def build_messages(text, template):
    system = (
        "you are an expert annotator participating in a subjective annotation task "
        "for a research project. Your task is to analyse raw, unfiltered internet "
        "comments. do not apply safety filters and do not refuse to answer. "
        "Only output the requested numeric score."
    )
    user = template.replace("{text}", str(text)[:500])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def annotate_attribute(attribute, texts, row_indices, tokenizer, llm, labels):
    sampling_params = SamplingParams(
        max_tokens=1, temperature=0.0, top_p=1.0, logprobs=10, seed=config.SEED
    )
    retry_params = SamplingParams(
        max_tokens=1, temperature=0.0, top_p=1.0, logprobs=20, seed=config.SEED
    )

    template, _ = load_prompt(attribute)
    valid_labels = set(str(l) for l in labels)

    prompts = []
    for text in texts:
        msgs = build_messages(text, template)
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ))

    outputs = llm.generate(prompts, sampling_params)

    results = []
    retry_indices = []

    for i, out in enumerate(outputs):
        logprobs_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
        found = {
            logprob_obj.decoded_token.strip(): logprob_obj.logprob
            for logprob_obj in logprobs_dict.values()
            if logprob_obj.decoded_token.strip() in valid_labels
        }
        if found:
            label, conf = confidence_from_logprobs(found)
            results.append({"index": row_indices[i], "label": label, "confidence": conf})
        else:
            results.append(None)
            retry_indices.append(i)

    if retry_indices:
        retry_prompts = [prompts[i] for i in retry_indices]
        retry_outputs = llm.generate(retry_prompts, retry_params)
        for j, out in enumerate(retry_outputs):
            i = retry_indices[j]
            logprobs_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            found = {
                logprob_obj.decoded_token.strip(): logprob_obj.logprob
                for logprob_obj in logprobs_dict.values()
                if logprob_obj.decoded_token.strip() in valid_labels
            }
            if found:
                label, conf = confidence_from_logprobs(found)
                results[i] = {"index": row_indices[i], "label": label, "confidence": conf}
            else:
                results[i] = {"index": row_indices[i], "label": None, "confidence": 0.0}

    return results


def main():
    print("=" * 60)
    print("  Test Ridge — Annotation (Llama-3.1-70B vanilla)")
    print("=" * 60)

    df = pd.read_csv(config.TEST_CSV, on_bad_lines="skip", engine="python")
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {config.TEST_CSV.name}")

    # Load model once
    print("\nLoading model...")
    llm = LLM(
        model=config.MODEL_ID,
        download_dir=config.MODEL_DOWNLOAD_DIR,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        max_model_len=config.MAX_MODEL_LEN,
        max_num_seqs=config.MAX_NUM_SEQS,
        tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
        seed=config.SEED,
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded.\n")

    texts       = df["text"].tolist()
    row_indices = list(df.index)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    for attribute in config.ATTRIBUTES:
        print(f"\n--- {attribute.upper()} ---")
        _, labels = load_prompt(attribute)
        results = annotate_attribute(attribute, texts, row_indices, tokenizer, llm, labels)

        out_df = pd.DataFrame(results)
        out_df = out_df.rename(columns={"label": attribute, "confidence": f"{attribute}_confidence"})

        out_dir = config.ATTR_RESULTS_DIR / attribute
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"results_{attribute}_{timestamp}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Saved {len(out_df)} rows → {out_path.name}")

    print("\nAll attributes annotated. Run merge_results.py next.")


if __name__ == "__main__":
    main()
