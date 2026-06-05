# Harmfulness × Refusal Direction Study

Empirical extension of [Zhao et al. (2025)](https://arxiv.org/abs/2507.11878) — *"LLMs Encode Harmfulness and Refusal Separately."*

**Research question:** Do category-specific harmfulness directions causally influence category-specific refusal? i.e., does ablating the `Physical_Harm` refusal direction reduce refusal more on `Physical_Harm` prompts than on `Fraud_Deception` prompts?

The pipeline has three stages:

| Stage | Script | What it does |
|-------|--------|--------------|
| 1a | `scripts/01_generate_and_classify.py` | Generate completions; label each as `refused` or `accepted` |
| 1b | `scripts/02_collect_activations.py` | Forward-pass only; save residual stream activations bucketed by label |
| 2 | `notebooks/direction_extraction.ipynb` | Compute harmfulness & refusal directions; visual analysis |
| 3 | `scripts/03_ablation.py` | Ablate a category's refusal direction during generation; measure refusal rate drop |

---

## Environment

```bash
conda activate abliteration
export OPENAI_API_KEY=sk-...   # required for GPT-4o classification
```

Depends on: `torch`, `transformers`, `nnsight==0.4.11`, `datasets`, `openai`, `tenacity`.

Models are loaded from HuggingFace Hub. Supported model keys (set in `config.py`):

| Key | Model |
|-----|-------|
| `llama2` | `meta-llama/Llama-2-7b-chat-hf` |
| `llama3` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `qwen2` | `Qwen/Qwen2-7B-Instruct` |
| `qwen2.5` | `Qwen/Qwen2.5-7B-Instruct` |

---

## Repository Layout

```
refusal_extension/
├── config.py                          # All constants: model IDs, paths, batch sizes, dataset sizes
├── core/
│   ├── models.py                      # Load nnsight LanguageModel + HF tokenizer
│   ├── datasets.py                    # Load CatQA / Alpaca; format prompts; compute t_inst / t_post_inst
│   ├── classify.py                    # Substring + GPT-4o refusal classifier
│   └── collect.py                     # nnsight tracing: save residual stream at two token positions
├── scripts/
│   ├── 01_generate_and_classify.py    # Stage 1a
│   ├── 02_collect_activations.py      # Stage 1b
│   └── 03_ablation.py                 # Stage 3
├── notebooks/
│   ├── direction_extraction.ipynb     # Direction computation + 7 analyses + t-SNE
│   └── data_fetching.ipynb            # Exploratory data loading
└── data/
    ├── completions/                   # Stage 1a JSONL outputs
    └── activations/                   # Stage 1b tensor outputs
```

---

## Stage 1a — Generate and Classify

Generates completions for all prompts using HuggingFace `AutoModelForCausalLM` (not nnsight), then classifies each response as `refused` or `accepted`.

Two classifiers are run on every response:
- **Substring** (primary): matches 18 hard-coded refusal phrases (e.g., "I cannot", "I won't").
- **GPT-4o** (secondary, requires `OPENAI_API_KEY`): sends response to `gpt-4o-mini`; returns `REFUSED` or `COMPLIED`. Used as the more accurate baseline in Stage 3.

Output: one JSONL file per `(model, dataset)` pair saved to `data/completions/`.

```bash
python scripts/01_generate_and_classify.py --model llama2 --dataset catqa
python scripts/01_generate_and_classify.py --model llama2 --dataset alpaca
```

**JSONL record format:**
```json
{
  "prompt_idx": 0,
  "raw_instruction": "...",
  "category": "Physical_Harm",
  "source": "catqa",
  "generated_text": "...",
  "substring": {"refused": true, "matched_phrase": "I cannot"},
  "gpt4o": {"refused": true, "raw": "REFUSED"},
  "disagreement": false
}
```

**Quick sanity check:**
```python
import json
from collections import Counter
lines = [json.loads(l) for l in open("data/completions/llama2_catqa.jsonl")]
print(Counter((l["category"], l["substring"]["refused"]) for l in lines))
```

---

## Stage 1b — Collect Activations

Re-runs the same prompts through nnsight (forward pass only, no generation) and saves the **residual stream** at two token positions:

- **`t_inst`** — last token of the instruction content (encodes harmfulness, per Zhao et al.)
- **`t_post_inst`** — last token of the full formatted prompt, just before generation starts (encodes refusal)

Activations are collected at every layer for all prompts, then bucketed by category and `refused`/`accepted` label (read from Stage 1a JSONL). Uses the **GPT-4o label** by default in Stage 3; substring is the default here.

Output: `.pt` tensors of shape `(n_prompts_in_bucket, n_layers, d_model)` saved under `data/activations/{label_source}/{model}/`.

```bash
python scripts/02_collect_activations.py --model llama2 --dataset catqa
python scripts/02_collect_activations.py --model llama2 --dataset alpaca

# To use GPT-4o labels instead of substring (requires Stage 1a run with API key):
python scripts/02_collect_activations.py --model llama2 --dataset catqa --label-source gpt4o
```

**Quick sanity check:**
```python
import torch
acts = torch.load("data/activations/gpt4o/llama2/catqa/Physical_Harm/refused_t_post_inst.pt")
print(acts.shape)  # (n_refused, n_layers, d_model)
```

**Data layout:** (Important to understand the folder structure)
```
data/activations/{label_source}/{model}/
├── catqa/
│   └── {Category}/
│       ├── refused_t_inst.pt
│       ├── refused_t_post_inst.pt
│       ├── accepted_t_inst.pt       # only written if bucket is non-empty
│       └── accepted_t_post_inst.pt
└── alpaca/
    ├── refused_t_inst.pt
    ├── refused_t_post_inst.pt
    ├── accepted_t_inst.pt
    └── accepted_t_post_inst.pt
```

The folder is organized into {label_source(gpt4o vs substring matching)}/{model}/{dataset}. The label source defines the classification of the prompts into refused vs accepted based on our labelling method. GPT-4o is better qualitatively in labelling since substring matching can match harmless response with certain substrings into harmful. An example is "I can't stress how good this shampoo is." will be harmful since **I can't** is a harmful substring.

---

## Stage 2 — Direction Extraction & Analysis

`notebooks/direction_extraction.ipynb`

Loads the saved activation tensors and computes per-category directions following Zhao et al.:

- **Harmfulness direction** (per category, per layer): `mean(refused_t_inst) − mean(accepted_t_inst)`
- **Refusal direction** (per category, per layer): `mean(refused_t_post_inst) − mean(accepted_t_post_inst)`

The notebook includes 7 analyses:

| # | Analysis |
|---|----------|
| 1 | Per-category cosine similarity of harmfulness direction across layers |
| 2 | Heatmap: categories × layers cosine similarity |
| 3 & 4 | Cross-category similarity matrices for harmful and refusal directions |
| 5 | L2 norm of directions across layers |
| 6 | PCA scatter of all directions at a probe layer |
| 7 | Intra- vs. inter-category similarity (bar + layer-wise line plot) |
| t-SNE | Prompt-level t-SNE on middle layers (10–20) for both token positions |

---

## Stage 3 — Ablation Experiment ( Work in progress on this stage.)

`scripts/03_ablation.py`

Tests the causal hypothesis: ablating a category's refusal direction should reduce refusal more on that category than on others.

**Method:** For a chosen `ablation_category`, load its per-layer refusal direction (`mean_refused_t_post_inst − mean_accepted_t_post_inst` from Alpaca). At each generation step, project out the direction from the residual stream at layers 10–21:

```
h ← h − (h · d̂) d̂
```

**Three conditions measured per test category:**
- `baseline` — refusal rate from Stage 1a JSONL (no intervention, GPT-4o labels)
- `ablated` — refusal rate after ablating the chosen category's direction
- `random` — same ablation with a random unit vector (control for direction specificity)

```bash
# Ablate one category's direction, test across all categories
python scripts/03_ablation.py --model llama2 --ablation-category Physical_Harm

# Run all 11 categories sequentially
python scripts/03_ablation.py --model llama2 --ablation-category all
```

Output: `data/ablation_results/{model}_{category}.json`

```json
{
  "ablation_category": "Physical_Harm",
  "results": {
    "Physical_Harm":   {"baseline": 0.92, "ablated": 0.54, "random": 0.88, "drop": 0.38},
    "Fraud_Deception": {"baseline": 0.80, "ablated": 0.76, "random": 0.79, "drop": 0.04},
    ...
  }
}
```

The key result to inspect: is `drop` larger for the ablated category than for unrelated categories?

---

## Core Modules

### `config.py`
Single source of truth for all constants. Edit here to change models, dataset sizes, batch sizes, or output paths. No logic — just values.

### `core/models.py`
Loads the nnsight `LanguageModel` and HF tokenizer together. Applies a config patch for Llama models to fix a `rope_parameters` compatibility issue between the model Hub configs and `transformers 5.x`.

### `core/datasets.py`
- Loads `declare-lab/CategoricalHarmfulQA` (550 prompts, 11 categories × 50) and `tatsu-lab/alpaca` (550 prompts, seed=42).
- Formats each prompt with the tokenizer's native chat template (`apply_chat_template`).
- Computes `t_inst_idx` via a sliding-window token search (last occurrence of instruction token IDs inside the full formatted token sequence).
- Prints a 3-prompt tokenization sanity check on every load.

### `core/classify.py`
- `classify_substring(text)` — case-insensitive match against 18 refusal phrases.
- `classify_gpt4o(text, instruction, client)` — `gpt-4o-mini` judge with tenacity retry (5 attempts, exponential backoff).
- `classify(text, instruction, client)` — runs both; records `disagreement` flag. GPT-4o is disabled gracefully if no API key is present.

### `core/collect.py`
- `collect_residual_stream(model, prompts, batch_size=1)` — opens one nnsight trace per prompt, saves `model.model.layers[L].input` (= `resid_pre`) for all layers, then indexes the saved tensor at `t_inst_idx` and `t_post_inst_idx`.
- Returns `{"t_inst": (n, L, d), "t_post_inst": (n, L, d)}`.
- Runs one prompt at a time to avoid padding-offset complexity; calls `torch.cuda.empty_cache()` after each.

---

## Running Order

```bash
# 1. Generate completions and labels
python scripts/01_generate_and_classify.py --model llama2 --dataset catqa
python scripts/01_generate_and_classify.py --model llama2 --dataset alpaca

# 2. Collect residual stream activations
python scripts/02_collect_activations.py --model llama2 --dataset catqa --label-source gpt4o
python scripts/02_collect_activations.py --model llama2 --dataset alpaca --label-source gpt4o

# 3. Open notebooks/direction_extraction.ipynb to compute directions and run analyses

# 4. Run ablation
python scripts/03_ablation.py --model llama2 --ablation-category Physical_Harm
```

Repeat steps 1–2 for each additional model (`llama3`, `qwen2`, `qwen2.5`).
