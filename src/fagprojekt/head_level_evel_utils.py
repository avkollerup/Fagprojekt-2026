from fagprojekt.SVD import do_SVD
from fagprojekt.model import (get_kvq)
import torch
import torch.nn.functional as F
import math

def find_token_positions(tokenizer, messages, needle):
    # take messages (system + user + assistant format) and converts it into the exact token sequence the model sees
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,)

    # extract the full token sequence as a list
    input_ids = encoded["input_ids"][0].tolist()

    # tokenize just the needle string
    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]

    # sliding window looking for exact needle match over token IDs
    for i in range(len(input_ids) - len(needle_ids) + 1):
        if input_ids[i:i + len(needle_ids)] == needle_ids:
            return list(range(i, i + len(needle_ids)))

    return []


# Computes both attention weights and final attention output
def get_attention_output(query_head, key_head, value_head):
    # Define dimensions
    d = key_head.shape[-1]
    T = query_head.shape[0]

    # Create mask to prevent attending to future tokens
    mask = torch.triu(torch.ones(T, T, device=query_head.device), diagonal=1)
    M = mask.masked_fill(mask == 1, float("-inf"))

    # Compute attention weights A = softmax(M + QK^T / sqrt(d))
    A = torch.softmax(M + (query_head @ key_head.T) / math.sqrt(d), dim=-1)

    # Compute attention output O = A @ V
    O = A @ value_head

    return A, O


def evaluate_head(query_head, key_true, value_true, key_approx, value_approx, needle_positions):
    # Compute true attention weights and outputs using original K and V
    A_true, O_true = get_attention_output(query_head, key_true, value_true)

    # Compute attention weights and outputs using approximated K and V
    A_approx, O_approx = get_attention_output(query_head, key_approx, value_approx)

    # Use the last query token
    q_idx = -1

    # Sum attention placed on the needle tokens
    true_needle_attention = A_true[q_idx, needle_positions].sum()
    approx_needle_attention = A_approx[q_idx, needle_positions].sum()

    # Compare using cosine similarity
    cosine = F.cosine_similarity(
        O_true[q_idx].unsqueeze(0), # make 1D tensor to 2D 
        O_approx[q_idx].unsqueeze(0), # make 1D tensor to 2D 
        dim=-1,
    )

    print("True attention on needle:", true_needle_attention.item())
    print("Approx attention on needle:", approx_needle_attention.item())
    print("Output cosine similarity:", cosine.item())

    return A_true, A_approx, O_true, O_approx


def method_1_K(key_head, k=50):
    """Method 1: Decomposition of the key matrix only"""
    U_K, s_K, Vt_K = do_SVD(key_head)

    k_eff = min(k, s_K.shape[0])
    K = U_K[:, :k_eff] @ torch.diag(s_K[:k_eff]) @ Vt_K[:k_eff, :]

    return K

def find_needle_heads(model, tokenizer, messages, needle, top_k=20):
    # Find the token positions of the needle in the full model input
    needle_positions = find_token_positions(tokenizer, messages, needle)

    # Stop if the needle cannot be found, since the attention score would be meaningless
    if len(needle_positions) == 0:
        raise ValueError("Needle not found in tokenized input.")

    # Store the attention score for each layer/head pair
    results = []

    # Number of transformer layers and KV heads to scan
    n_layers = len(model.model.layers)
    n_heads = model.config.num_key_value_heads

    # Loop through every layer and KV head
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):

            # Extract Q, K and V for the current layer/head
            key_head, value_head, query_head = get_kvq(
                messages,
                layer_idx=layer_idx,
                head_idx=head_idx,
                want_print=False,
                model=model,
                tokenizer=tokenizer,
            )

            # Compute attention weights and attention output for this head
            A, O = get_attention_output(query_head, key_head, value_head)

            # Use the last token as the query token
            q_idx = -1

            # Total attention mass placed on all needle tokens
            needle_attention = A[q_idx, needle_positions].sum().item()

            # Largest single-token attention value for this query
            max_attention = A[q_idx].max().item()

            # Average attention over all tokens for this query
            mean_attention = A[q_idx].mean().item()

            # Save scores for this layer/head
            results.append({
                "layer": layer_idx,
                "head": head_idx,
                "needle_attention": needle_attention,
                "max_attention": max_attention,
                "mean_attention": mean_attention,
            })

    # Sort heads by how much attention they place on the needle
    results = sorted(
        results,
        key=lambda x: x["needle_attention"],
        reverse=True,
    )

    # Print the best scoring heads
    print("\n--- TOP NEEDLE HEADS ---")
    for r in results[:top_k]:
        print(
            f"layer={r['layer']:2d}, head={r['head']:2d} | "
            f"needle_attn={r['needle_attention']:.6f} | "
            f"max_attn={r['max_attention']:.6f} | "
            f"mean_attn={r['mean_attention']:.6f}"
        )

    return results