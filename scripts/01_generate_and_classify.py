"""Stage 1a: generate completions and classify refusal behavior.

Usage:
    python scripts/01_generate_and_classify.py --model llama2 --dataset catqa
    python scripts/01_generate_and_classify.py --model llama2 --dataset alpaca
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.datasets import load_catqa, load_alpaca
from core.classify import classify


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, choices=list(config.MODELS))
    p.add_argument("--dataset", required=True, choices=["catqa", "alpaca"])
    return p.parse_args()


def load_hf_model(model_key):
    model_id  = config.MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, batch_prompts):
    formatted = [p["formatted"] for p in batch_prompts]
    enc = tokenizer(formatted, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False)
    input_len = enc["input_ids"].shape[1]
    return [
        tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
        for i in range(len(batch_prompts))
    ]


def main():
    args = parse_args()

    client = None
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key= api_key)
    else:
        print("WARNING: OPENAI_API_KEY not set — GPT-4o classifier disabled, substring labels only.")

    model, tokenizer = load_hf_model(args.model)

    if args.dataset == "catqa":
        prompts = load_catqa(tokenizer, args.model)
    else:
        prompts = load_alpaca(tokenizer, args.model)

    os.makedirs(config.COMPLETIONS_DIR, exist_ok=True)
    out_path = os.path.join(config.COMPLETIONS_DIR, f"{args.model}_{args.dataset}.jsonl")

    records = []
    for start in range(0, len(prompts), config.GEN_BATCH_SIZE):
        batch = prompts[start : start + config.GEN_BATCH_SIZE]
        texts = generate_batch(model, tokenizer, batch)
        for i, (p, text) in enumerate(zip(batch, texts)):
            result = classify(text, p["raw_instruction"], client)
            records.append({
                "prompt_idx":      start + i,
                "raw_instruction": p["raw_instruction"],
                "category":        p["category"],
                "source":          p["source"],
                "generated_text":  text,
                "substring":       result["substring"],
                "gpt4o":           result["gpt4o"],
                "disagreement":    result["disagreement"],
            })
        print(f"  {start + len(batch)}/{len(prompts)} prompts processed")

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nSaved {len(records)} records → {out_path}")
    print("\nRefusal rate by category (substring):")
    cat_counts = defaultdict(lambda: {"total": 0, "refused": 0})
    for r in records:
        cat = r["category"] or "alpaca"
        cat_counts[cat]["total"]   += 1
        cat_counts[cat]["refused"] += int(r["substring"]["refused"])
    for cat, v in sorted(cat_counts.items()):
        pct = 100 * v["refused"] / v["total"] if v["total"] else 0
        print(f"  {cat}: {v['refused']}/{v['total']} ({pct:.0f}%)")
    disagree = sum(r["disagreement"] for r in records)
    print(f"\nDisagreements (substring vs gpt4o): {disagree}/{len(records)}")


if __name__ == "__main__":
    main()
