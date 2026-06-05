"""Stage 1b: collect residual stream activations and save bucketed tensors.

Reads the JSONL from Stage 1a for refusal labels. Uses the substring label
by default (the primary classifier). Pass --label-source gpt4o to use GPT-4o
labels instead (requires that Stage 1a was run with OPENAI_API_KEY set).

Usage:
    python scripts/02_collect_activations.py --model llama2 --dataset catqa
    python scripts/02_collect_activations.py --model llama2 --dataset alpaca
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.models import load_model
from core.datasets import load_catqa, load_alpaca
from core.collect import collect_residual_stream


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True, choices=list(config.MODELS))
    p.add_argument("--dataset",      required=True, choices=["catqa", "alpaca"])
    p.add_argument("--label-source", default="substring", choices=["substring", "gpt4o"])
    return p.parse_args()


def load_labels(model_key, dataset, label_source):
    path = os.path.join(config.COMPLETIONS_DIR, f"{model_key}_{dataset}.jsonl")
    records = [json.loads(line) for line in open(path)]
    labels = []
    for r in records:
        if label_source == "gpt4o":
            if r["gpt4o"] is None:
                raise ValueError(
                    f"gpt4o label is None for prompt {r['prompt_idx']}. "
                    "Re-run Stage 1a with OPENAI_API_KEY set."
                )
            labels.append(r["gpt4o"]["refused"])
        else:
            labels.append(r["substring"]["refused"])
    return labels

def format_category(category_name):
    return category_name.replace(" ","_").replace("/","_")

def save_bucket(acts, out_dir, label, position):

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}_{position}.pt")
    torch.save(acts, path)
    print(f"  Saved {tuple(acts.shape)} → {path}")


def main():
    args = parse_args()
    print(f"Label source: {args.label_source}")

    model, tokenizer = load_model(args.model)

    if args.dataset == "catqa":
        prompts = load_catqa(tokenizer, args.model)
    else:
        prompts = load_alpaca(tokenizer, args.model)

    labels = load_labels(args.model, args.dataset, args.label_source)
    assert len(labels) == len(prompts), (
        f"Label count {len(labels)} != prompt count {len(prompts)}"
    )

    print(f"\nCollecting activations for {len(prompts)} prompts...")
    acts = collect_residual_stream(model, prompts, batch_size=config.COLLECT_BATCH_SIZE)

    label_name = {True: "refused", False: "accepted"}

    print("\nSaving buckets...")
    if args.dataset == "catqa":
        base_dir = os.path.join(config.ACTIVATIONS_DIR, args.label_source, args.model, "catqa")
        for cat in config.CATQA_CATEGORIES:
            cat_idxs = [i for i, p in enumerate(prompts) if p["category"] == cat]
            for refused in [True, False]:
                bucket = [i for i in cat_idxs if labels[i] == refused]
                lname  = label_name[refused]
                if not bucket:
                    print(f"  SKIP (empty): catqa/{cat}/{lname}")
                    continue

                formatted_category = format_category(cat)
                out_dir = os.path.join(base_dir, formatted_category)
                for pos in ["t_inst", "t_post_inst"]:
                    save_bucket(acts[pos][bucket], out_dir, lname, pos)
    else:
        base_dir = os.path.join(config.ACTIVATIONS_DIR, args.label_source, args.model, "alpaca")
        for refused in [True, False]:
            bucket = [i for i, r in enumerate(labels) if r == refused]
            lname  = label_name[refused]
            if not bucket:
                print(f"  SKIP (empty): alpaca/{lname}")
                continue
            for pos in ["t_inst", "t_post_inst"]:
                save_bucket(acts[pos][bucket], base_dir, lname, pos)

    print("\nDone.")


if __name__ == "__main__":
    main()
