from fagprojekt.model import (
    load_model,
    get_response,
    extract_KV,
    extract_query,
    # compute_attention_weights,
)
import torch
import numpy as np
from torch.linalg import svd
import matplotlib.pyplot as plt

model, tokenizer = load_model()

messages = [
    {"role": "system", "content": "You are a helpful assistant"}, # Besked til modellen om hvordan den skal opføre sig
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"}, # Besked fra user (os)
]
inputs, outputs, generated_tokens = get_response(model, tokenizer, messages)

print("-------------- SYSTEM PROMPT AND REPLY --------------")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))

input_length = inputs.input_ids.shape[1]
print("-------------- REPLY ONLY --------------")
print(tokenizer.decode(generated_tokens[input_length:], skip_special_tokens=True))

print("-------------- KV CACHE --------------")
layer_idx = 0
head_idx = 0
KV_cache, key, value, key_head, value_head = extract_KV(outputs, layer_idx, head_idx)

query_inputs = {
    "input_ids": generated_tokens.unsqueeze(0),
    "attention_mask": torch.ones_like(generated_tokens).unsqueeze(0),
}
query, query_head = extract_query(model, query_inputs, layer_idx, head_idx)

kv_seq_len = key.shape[2]
if query.shape[2] != kv_seq_len:
    query = query[:, :, :kv_seq_len, :]
    query_head = query_head[:kv_seq_len, :]

print(f"{len(KV_cache)} transformer layers in the model")

print(f"KV cache from transformer layer {layer_idx}:")
print(f"K dimension: {key.shape}")
print(f"V dimension: {value.shape}\n")

print(f"KV cache from transformer layer {layer_idx} and head {head_idx}:")
print(f"Key dimension: {key_head.shape}")
print(f"Value dimension: {value_head.shape}")
print(f"Query dimension: {query_head.shape}\n")

# true_attn_weights = compute_attention_weights(model, query, key)
# true_attn_head = true_attn_weights[0, head_idx]
# print(f"Attention weights dimension: {true_attn_weights.shape}\n")

def do_SVD(matrix, name, plot=False):
    # CUDA SVD does not support half precision; upcast for decomposition.
    svd_input = matrix.to(torch.float32)
    U, s, Vh = svd(svd_input)

    if plot:
        plt.figure()
        plt.plot(np.cumsum((s**2/sum(s**2)).detach().cpu().numpy()))
        plt.title(f'Explained variance by principal components of {name}')
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.savefig(f'reports/figures/explained_var_{name}.pdf')
    return U, s, Vh

# Method 1: Decomposition of the key matrix only
U_K, s_K, Vh_K = do_SVD(key_head,"K", True)

# Truncate and calculate K again
k = 50
K = U_K[:,:k] @ torch.diag(s_K)[:k,:k] @ Vh_K[:,:k].T
print(K.shape)
M = torch.tril(torch.ones((K.shape[0], K.shape[0]), device=K.device, dtype=K.dtype))
attn_weights_K = torch.softmax((M + query_head.to(torch.float32)) @ K.T, dim=-1) @ value_head.to(torch.float32)


# Method 2:
U_V, s_V, Vh_V = do_SVD(value_head,"V", True)

