"""Stage 3: Ablation experiment.

For a chosen ablation_category, ablates its refusal direction from the
residual stream at layers 10–21 during generation and records the GPT-4o
refusal rate across all CatQA categories. Compares against:
  - baseline : refusal rates read from the Stage 1a JSONL (no intervention)
  - random   : same ablation with a random unit vector (control)

Usage:
    python scripts/03_ablation.py --model llama2 --ablation-category Physical_Harm
    python scripts/03_ablation.py --model llama2 --ablation-category all
"""
import argparse
import json
import os
import sys

import torch
from openai import OpenAI

import warnings
warnings.filterwarnings('ignore')

from transformers import logging

# Only show errors
logging.set_verbosity_error()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.models import load_model
from core.datasets import load_catqa
from core.classify import classify_gpt4o

ABLATE_LAYERS = list(range(10, 22))   # layers 10–21 inclusive
LABEL_SOURCE  = "gpt4o"

# Map config category names ("Physical Harm") → directory names ("Physical_Harm")
def fmt(cat):
    return cat.replace(" ", "_").replace("/", "_")

# Reverse map for prompts filtering: "Physical_Harm" → "Physical Harm"
FMT_TO_RAW = {fmt(c): c for c in config.CATQA_CATEGORIES}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(config.MODELS))
    p.add_argument(
        "--ablation-category", required=True,
        help="Formatted category name (e.g. Physical_Harm) or 'all'",
    )
    return p.parse_args()


def load_refusal_direction(model_key, fmt_category):
    """Compute per-layer refusal direction from saved activations. Returns (n_layers, d_model)."""
    base     = os.path.join(config.ACTIVATIONS_DIR, LABEL_SOURCE, model_key)
    refused  = torch.load(
        os.path.join(base, "catqa", fmt_category, "refused_t_post_inst.pt"),
        weights_only=True,
    )   # (n_refused, n_layers, d_model)
    accepted = torch.load(
        os.path.join(base, "alpaca", "accepted_t_post_inst.pt"),
        weights_only=True,
    )   # (n_accepted, n_layers, d_model)
    return refused.mean(dim=0) - accepted.mean(dim=0)   # (n_layers, d_model)


def unit(v):
    return v / v.norm().clamp(min=1e-8)


def generate_ablated(model, tokenizer, prompt, layer_units):
    """
    Generate with direction ablation applied at every autoregressive step.
    layer_units: {L: unit_vector (d_model,)} on CPU.
    Ablation: h ← h − (h·d̂)d̂  at each of ABLATE_LAYERS.
    .all() on each layer Envoy makes the hook fire at every generation step,
    not just the first forward pass.
    """
    with model.generate(prompt, max_new_tokens=config.MAX_NEW_TOKENS) as tracer:
        out = model.generator.output.save()
        for L in ABLATE_LAYERS:
            d = layer_units[L].to(config.DEVICE)
            with model.model.layers[L].all():
                h = model.model.layers[L].output[0]        # hidden states (1, seq, d_model)
                proj = (h * d).sum(dim=-1, keepdim=True) * d
                model.model.layers[L].output[0] = h - proj

    input_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def gpt4o_refusal_rate(pairs, client):
    """pairs: list of (generated_text, instruction). Returns fraction refused."""
    n = sum(classify_gpt4o(text, instr, client)["refused"] for text, instr in pairs)
    return n / len(pairs)


def read_baseline_rates(model_key):
    """Read per-category GPT-4o refusal rates from Stage 1a JSONL (keyed by raw name)."""
    path    = os.path.join(config.COMPLETIONS_DIR, f"{model_key}_catqa.jsonl")
    records = [json.loads(l) for l in open(path)]
    rates   = {}
    for cat in config.CATQA_CATEGORIES:
        cat_recs = [r for r in records if r["category"] == cat and r.get("gpt4o")]
        rates[cat] = sum(r["gpt4o"]["refused"] for r in cat_recs) / len(cat_recs) \
                     if cat_recs else None
    return rates


def run_one(model, tokenizer, prompts, fmt_cat, baseline_rates, client):
    """Ablate fmt_cat's refusal direction; evaluate across all categories."""
    direction    = load_refusal_direction(args.model, fmt_cat)
    dir_units    = {L: unit(direction[L]) for L in ABLATE_LAYERS}
    random_units = {L: unit(torch.randn_like(direction[L])) for L in ABLATE_LAYERS}

    results = {}
    for raw_cat in config.CATQA_CATEGORIES:
        cat_prompts = [p for p in prompts if p["category"] == raw_cat]
        print(f"  [{fmt_cat} → {fmt(raw_cat)}]  {len(cat_prompts)} prompts ...")

        ablated = [
            (generate_ablated(model, tokenizer, p["formatted"], dir_units),
             p["raw_instruction"])
            for p in cat_prompts
        ]
        random_ctrl = [
            (generate_ablated(model, tokenizer, p["formatted"], random_units),
             p["raw_instruction"])
            for p in cat_prompts
        ]

        base = baseline_rates[raw_cat]
        abl  = gpt4o_refusal_rate(ablated, client)
        rnd  = gpt4o_refusal_rate(random_ctrl, client)

        results[fmt(raw_cat)] = {
            "baseline": base,
            "ablated":  round(abl, 4),
            "random":   round(rnd, 4),
            "drop":     round(base - abl, 4) if base is not None else None,
        }
        print(f"    baseline={base:.3f}  ablated={abl:.3f}  "
              f"random={rnd:.3f}  drop={results[fmt(raw_cat)]['drop']:.3f}")

    return results


def main():
    global args
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set — GPT-4o is used for classification."
    client = OpenAI(api_key=api_key)

    model, tokenizer = load_model(args.model)
    prompts          = load_catqa(tokenizer, args.model)
    baseline_rates   = read_baseline_rates(args.model)

    ablation_cats = list(FMT_TO_RAW.keys()) if args.ablation_category == "all" \
                    else [args.ablation_category]

    os.makedirs("data/ablation_results", exist_ok=True)

    for fmt_cat in ablation_cats:
        assert fmt_cat in FMT_TO_RAW, \
            f"Unknown category '{fmt_cat}'. Valid: {list(FMT_TO_RAW)}"
        print(f"\n{'='*60}\nAblating direction: {fmt_cat}")

        results  = run_one(model, tokenizer, prompts, fmt_cat, baseline_rates, client)
        out_path = f"data/ablation_results/{args.model}_{fmt_cat}.json"
        with open(out_path, "w") as f:
            json.dump({"ablation_category": fmt_cat, "results": results}, f, indent=2)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
