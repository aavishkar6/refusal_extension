import random
import torch
from datasets import load_dataset
import config


def find_t_inst(inst_ids, full_ids):
    """
    Sliding window: find the last element index of inst_ids as a contiguous
    subsequence in full_ids. Searches from the right so we match the correct
    occurrence when the instruction text repeats.
    """
    n = len(inst_ids)
    for i in range(len(full_ids) - n, -1, -1):
        if full_ids[i : i + n] == inst_ids:
            return i + n - 1
    raise ValueError(
        f"inst_ids subsequence not found in full_ids.\n"
        f"  inst ({len(inst_ids)} tokens): {inst_ids[:20]}\n"
        f"  full ({len(full_ids)} tokens): {full_ids[:20]}"
    )


def format_prompt(instruction, tokenizer, model_name):
    messages = [{"role": "user", "content": instruction}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(formatted, add_special_tokens=False)
    inst_ids = tokenizer.encode(instruction, add_special_tokens=False)

    # Some tokenizers prepend a space inside the template; try with leading space
    # if the plain encode doesn't match.
    try:
        t_inst_idx = find_t_inst(inst_ids, full_ids)
    except ValueError:
        inst_ids_spaced = tokenizer.encode(" " + instruction, add_special_tokens=False)
        t_inst_idx = find_t_inst(inst_ids_spaced, full_ids)

    t_post_inst_idx = len(full_ids) - 1

    return {
        "raw_instruction":  instruction,
        "formatted":        formatted,
        "input_ids":        torch.tensor(full_ids),
        "t_inst_idx":       t_inst_idx,
        "t_post_inst_idx":  t_post_inst_idx,
        "category":         None,
        "source":           None,
    }


def _print_sanity(prompts, tokenizer, n=3):
    print("\n--- Tokenization sanity check ---")
    for i, p in enumerate(prompts[:n]):
        ids = p["input_ids"].tolist()
        ti  = p["t_inst_idx"]
        tp  = p["t_post_inst_idx"]
        print(f"Prompt {i}:")
        print(f"  Full:          {repr(p['formatted'])}")
        print(f"  t_inst     (idx {ti}): {repr(tokenizer.decode([ids[ti]]))}")
        print(f"  t_post_inst(idx {tp}): {repr(tokenizer.decode([ids[tp]]))}")
    print("---------------------------------\n")


def load_catqa(tokenizer, model_name):
    ds = load_dataset("declare-lab/CategoricalHarmfulQA", split="en") # Get the english split.
    rng = random.Random(config.SEED)
    prompts = []
    for cat in config.CATQA_CATEGORIES:
        cat_rows = [r for r in ds if r["Category"] == cat]
        sampled  = rng.sample(cat_rows, min(config.N_CATQA_PER_CATEGORY, len(cat_rows)))
        for row in sampled:
            p = format_prompt(row["Question"], tokenizer, model_name)
            p["category"] = cat
            p["source"]   = "catqa"
            prompts.append(p)
    _print_sanity(prompts, tokenizer)
    return prompts


def load_alpaca(tokenizer, model_name, n=None):
    if n is None:
        n = config.N_ALPACA
    ds  = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(config.SEED)
    # filter out rows with empty instructions
    rows = [r for r in ds if r["instruction"].strip()]
    rows = rng.sample(rows, n)
    prompts = []
    for row in rows:
        p = format_prompt(row["instruction"], tokenizer, model_name)
        p["category"] = None
        p["source"]   = "alpaca"
        prompts.append(p)
    _print_sanity(prompts, tokenizer)
    return prompts
