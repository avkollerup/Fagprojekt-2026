# imports 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from fagprojekt.SVD import do_SVD
from fagprojekt.model import (
load_model,
get_kvq,
get_messages,
)

from collections import defaultdict
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

print("hej")

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

#------------------------EVALUATION--------------------------
# load model only once 
model,tokenizer = load_model()

# Choose the document containing the needle
path = f'document-haystack/AIG/AIG_10Pages/Text_TextNeedles/AIG_10Pages_TextNeedles_page_{1}.txt'

# Create the chat messages
messages, prompt, needle = get_messages(path, num_tokens=100)
print(messages)

# Add the true needle answer to the input, so we can inspect attention from the answer tokens (already there?)
# messages[1]["content"] += " " + needle

# Extract K, V, and Q
key_head, value_head, query_head = get_kvq(
    messages,
    layer_idx=0,
    head_idx=0,
    want_print=False,
    model=model,
    tokenizer=tokenizer,
)

# Find the exact token positions where the needle appears in the full model input
needle_positions = find_token_positions(tokenizer, messages, needle)
print("Needle:", needle)
print("Needle positions:", needle_positions)

# Approximate the key matrix using SVD method 1
key_approx = method_1_K(key_head)

# Keep V unchanged for this first test, so we only test the effect of approximating K
value_approx = value_head

# Compare true attention/output with approximated attention/output
A_true, A_approx, O_true, O_approx = evaluate_head(
    query_head,
    key_head,
    value_head,
    key_approx,
    value_approx,
    needle_positions,
)

# Plot the attention of the last query token using the true KV-cache
plt.plot(A_true[-1].detach().cpu(), label="True attention")

# Plot the attention of the last query token using the KV-cache with the approximated K
plt.plot(A_approx[-1].detach().cpu(), label="Approx attention")

# Highlight where in the plot where the needle tokens are located
plt.axvspan(needle_positions[0], needle_positions[-1], alpha=0.2)

plt.legend()
plt.savefig("reports/figures/eval.png", dpi=150)
plt.close()

