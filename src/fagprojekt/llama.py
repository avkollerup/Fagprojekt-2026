print('started')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import math
import seaborn as sns
import numpy as np
from fagprojekt.model import load_model, extract_KV, extract_query
from fagprojekt.head_level_eval_utils import find_token_positions, get_random_messages
import matplotlib.pyplot as plt

def extract_full_kv(model, inputs, layer_idx, head_idx):
    """Extract full key and value tensors for one layer/head from the full input sequence."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        forward_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    layer_module = model.model.layers[layer_idx]
    hidden_in = forward_outputs.hidden_states[layer_idx]
    normed_hidden = layer_module.input_layernorm(hidden_in)

    key = layer_module.self_attn.k_proj(normed_hidden)
    value = layer_module.self_attn.v_proj(normed_hidden)

    batch_size, seq_len, embed_dim = key.shape
    num_heads = getattr(layer_module.self_attn, "num_heads", None)
    head_dim = getattr(layer_module.self_attn, "head_dim", None)

    if num_heads is None and head_dim is not None:
        num_heads = embed_dim // head_dim
    elif head_dim is None and num_heads is not None:
        head_dim = embed_dim // num_heads
    elif num_heads is None and head_dim is None:
        head_dim = 128
        num_heads = embed_dim // head_dim

    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    key_head = key[0][head_idx]
    value_head = value[0][head_idx]
    return key, key_head, value, value_head


def compute_attention_weights(query_head, key_head, head_dim=128):
    """Compute attention weights from Q and K tensors with causal mask.
    
    Args:
        query_head: (seq_len, head_dim)
        key_head: (seq_len, head_dim)
        head_dim: dimension of attention head
    
    Returns:
        attention weights: (seq_len, seq_len)
    """
    # Compute attention scores
    scores = (query_head @ key_head.T) / math.sqrt(head_dim)
    
    # Apply causal mask for possibly non-square query/key lengths
    seq_len_q = query_head.shape[0]
    seq_len_k = key_head.shape[0]
    mask = torch.triu(
        torch.full((seq_len_q, seq_len_k), float("-inf"), device=query_head.device, dtype=query_head.dtype),
        diagonal=1,
    )
    scores = scores + mask
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    return attn_weights


def main():
    print("running")
    model, tokenizer = load_model(want_print=True)
    model.eval()

    layer_idx = 5
    head_idx = 0
    local_window = 128

    #path = "document-haystack/BankOfMontreal/BankOfMontreal_25Pages/Text_TextNeedles/BankOfMontreal_25Pages_TextNeedles_page_9.txt"
    path = "document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_3.txt"
    num_tokens = 400  

    messages, _, needle = get_random_messages(path, num_tokens=num_tokens, local_window_size=local_window)

    # prompt = 'What is the secret color?'
    # needle = " The secret color is green"
    # text = ' ' * 50 + needle + ' x' * 150  
    # messages = [
    #     {"role": "system", "content": "You will recieve a question of the form 'What is the secret (key) in the document?' and must answer in the form 'The secret (key) is (value)'."},
    #     {"role": "user", "content": f"Read the following text and answer the question: {prompt},{text}"},
    # ]

    # prompt = 'What is the secret color?'
    # needle = " The secret color is green"
    # text = f' x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x  x x x x x x x x x x x x x x x x x x x x{needle} x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x '
    # # overskriv med eksempel for at undgå støj i attention
    # messages = [
    # {"role": "system", "content": "You will recieve a question of the form 'What is the secret (key) in the document?' and must answer in the form 'The secret (key) is (value).'."}, # Besked til modellen om hvordan den skal opføre sig
    # {"role": "user", "content": f"Read the following text and answer the question: {prompt},{text}"}, # Besked fra user (os) 
    # ]


    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)
    
    needle_pos = find_token_positions(tokenizer, messages, needle)
    print(f"Needle token positions: {needle_pos}")

    # Generate with KV cache - NO output_attentions to save memory
    torch.cuda.empty_cache()
    with torch.no_grad():
        generation_outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # Reduced from 21
            return_dict_in_generate=True,
            use_cache=True,
            output_attentions=False,
            output_scores=False
        )

    generated_ids = generation_outputs.sequences
    
    # Get query from the full generated sequence
    query_inputs = {
        "input_ids": generated_ids,
        "attention_mask": torch.ones_like(generated_ids),
    }
    query, query_head = extract_query(model, query_inputs, layer_idx, head_idx)

    # Extract full keys/values for the same full generated sequence so generated tokens are covered
    _, key_head, _, value_head = extract_full_kv(model, query_inputs, layer_idx, head_idx)

    # Convert to float32 for numerical stability and move to CPU
    query_head = query_head.to(torch.float32).cpu()
    key_head = key_head.to(torch.float32).cpu()

    # Compute attention weights directly from Q @ K^T
    head_dim = query_head.shape[-1]
    full_attn = compute_attention_weights(query_head, key_head, head_dim=head_dim)
    
    print(f"Attention matrix shape: {full_attn.shape}")

    seq_len = generated_ids.shape[-1]
    print(f"Length of input tokens: {seq_len}")
    print("\n--- MODEL OUTPUT ---")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    
    needle_min = min(needle_pos)
    needle_max = max(needle_pos)
    needle_color = "darkgreen"

    print("\n--- ATTENTION MATRIX ---")
    
    attn_matrix = full_attn
    value_range = torch.quantile(torch.abs(attn_matrix), 0.995).item()
    if value_range < 1e-4:
        value_range = 1e-4

    fig, ax_heat = plt.subplots(figsize=(12, 10))

    sns.heatmap(attn_matrix.detach().numpy(), cmap="vlag", vmin=-value_range, vmax=value_range, cbar_kws={"label": "attention value"}, ax=ax_heat)
    ax_heat.set_title(f"Attention Heatmap for Head {head_idx}, Layer {layer_idx}")
    ax_heat.set_xlabel("Key Position")
    ax_heat.set_ylabel("Query Position")

    ax_heat.axvline(seq_len, color="gray", linestyle="--", linewidth=1)
    ax_heat.axhline(seq_len, color="gray", linestyle="--", linewidth=1)

    ax_heat.plot([needle_min + 0.5, needle_max + 0.5], [1.02, 1.02], color=needle_color, linewidth=3, transform=ax_heat.get_xaxis_transform(), clip_on=False)
    ax_heat.text((needle_min + needle_max) / 2 + 0.5, 1.05, "needle", color=needle_color, ha="center", va="bottom", transform=ax_heat.get_xaxis_transform())

    ax_heat.plot([-0.08, -0.08], [needle_min + 0.5, needle_max + 0.5], color=needle_color, linewidth=3, transform=ax_heat.get_yaxis_transform(), clip_on=False)
    ax_heat.text(-0.08, (needle_min + needle_max) / 2 + 0.5, "needle", color=needle_color, ha="right", va="center", transform=ax_heat.get_yaxis_transform())

    plt.savefig(f"reports/figures/attention_heatmaps/causal_llama_AIG_attention_heatmap_head{head_idx}.png")
    plt.close()


if __name__ == "__main__":
    main()

