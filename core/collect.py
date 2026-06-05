import torch
import config


def collect_residual_stream(model, prompts, batch_size=1):
    """
    Collect resid_pre at t_inst and t_post_inst for every layer, every prompt.

    Returns:
        {
            "t_inst":      tensor (n_prompts, n_layers, d_model),
            "t_post_inst": tensor (n_prompts, n_layers, d_model),
        }

    batch_size=1 is the safe default — avoids all padding-index arithmetic.
    Increase only after verifying that the offset correction for left-padded
    batches is correct.

    resid_pre is the hidden state *arriving* at layer L, equivalent to
    TransformerLens's resid_pre hook point. In nnsight on Llama/Qwen2,
    model.model.layers[L].input is the Envoy alias for the first positional
    argument to the layer's forward call (i.e. hidden_states before that layer
    transforms it).

    When we add MLP-output or attn-output collection in Phase 2, we add a
    separate collect_mlp_outputs() with the same signature — we do not modify
    this function.
    """
    n_layers = model.config.num_hidden_layers
    t_inst_list      = []
    t_post_inst_list = []

    for idx, prompt in enumerate(prompts):
        if idx % 10 == 0:
            print(f"  Collecting {idx}/{len(prompts)} ...")

        formatted = prompt["formatted"]
        ti = prompt["t_inst_idx"]
        tp = prompt["t_post_inst_idx"]

        layer_saves = []

        with torch.no_grad():
            with model.trace(formatted):
                for L in range(n_layers):
                    # .input is the Envoy alias for inputs[0][0], which is
                    # hidden_states arriving at this layer — the resid_pre.
                    # Shape inside the trace graph: (1, seq_len, d_model).
                    saved = model.model.layers[L].input.save()
                    layer_saves.append(saved)

        # Values are real tensors once the trace context exits.
        t_inst_vecs      = []
        t_post_inst_vecs = []
        for L in range(n_layers):
            act = layer_saves[L].value  # (1, seq_len, d_model)
            t_inst_vecs.append(act[0, ti, :].cpu())
            t_post_inst_vecs.append(act[0, tp, :].cpu())

        t_inst_list.append(torch.stack(t_inst_vecs, dim=0))       # (n_layers, d_model)
        t_post_inst_list.append(torch.stack(t_post_inst_vecs, dim=0))

        torch.cuda.empty_cache()

    return {
        "t_inst":      torch.stack(t_inst_list,      dim=0),  # (n_prompts, n_layers, d_model)
        "t_post_inst": torch.stack(t_post_inst_list, dim=0),
    }
