import torch
from nnsight import LanguageModel
from transformers import AutoConfig, AutoTokenizer
import config


def _patch_rope_config(hf_config):
    """
    Transformers 5.x requires a 'factor' key whenever rope_type='llama3',
    but older Llama-2 and some Llama-3 Hub configs were uploaded without it.
    Patching in-place before the model is instantiated avoids the KeyError.
    """
    rope = getattr(hf_config, "rope_parameters", None)
    if not rope or not isinstance(rope, dict):
        return
    rope_type = rope.get("rope_type", rope.get("type", ""))
    # if rope_type == "llama3" and "factor" not in rope:
    rope["factor"] = 8.0
    rope["low_freq_factor"] = 1.0
    rope["high_freq_factor"] = 4.0


def load_model(model_key):
    """
    Returns (nnsight_model, hf_tokenizer).

    We load the HF tokenizer separately so datasets.py can use it before or
    without loading the full nnsight model (useful for Stage 1a which uses
    plain HF generate instead of nnsight).
    """
    model_id  = config.MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and patch config before passing to nnsight so that transformers 5.x
    # doesn't blow up on Llama models whose Hub configs lack the 'factor' key.
    hf_config = AutoConfig.from_pretrained(model_id)
    # print(hf_config)
    _patch_rope_config(hf_config)

    model = LanguageModel(
        model_id,
        config=hf_config,
        dispatch=True,
        device_map=config.DEVICE,
        dtype=torch.bfloat16,
    )
    return model, tokenizer
