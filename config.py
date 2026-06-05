MODELS = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen2":  "Qwen/Qwen2-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct"
}

DATA_DIR        = "data"
COMPLETIONS_DIR = "data/completions"
ACTIVATIONS_DIR = "data/activations"

CATQA_CATEGORIES = [
    'Adult Content',
    'Child Abuse',
    'Economic Harm',
    'Fraud/Deception',
    'Hate/Harass/Violence',
    'Illegal Activity',
    'Malware Viruses',
    'Physical Harm',
    'Political Campaigning',
    'Privacy Violation Activity',
    'Tailored Financial Advice'
]

# Pilot scale — change to 50 / 550 for full run
N_CATQA_PER_CATEGORY = 50
N_ALPACA             = 550
SEED                 = 42

MAX_NEW_TOKENS  = 100
GEN_BATCH_SIZE  = 8

# nnsight tracing is more memory-intensive than HF generate
COLLECT_BATCH_SIZE = 1
DTYPE              = "bfloat16"
DEVICE             = "cuda"

GPT4O_MODEL      = "gpt-4o-mini"
GPT4O_CHAR_LIMIT = 400
