# imports 
from fagprojekt.SVD import method_1,method_2,method_3
from fagprojekt.model import (
load_model,
get_kvq,
get_messages,
get_true_attention_values,
extract_KV,
extract_query
)

from collections import defaultdict
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

def find_token_positions(tokenizer, messages, needle):
    # Convert the chat messages into the exact token sequence seen by the model
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()

    # Tokenize the needle
    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]

    # Slide a window over the full input to find an exact token match of the needle
    for i in range(len(input_ids) - len(needle_ids) + 1):
        if input_ids[i:i + len(needle_ids)] == needle_ids:
            # Return all token indices where the needle appears
            return list(range(i, i + len(needle_ids)))

    return []


# Computes both attention weights and final attention output
def get_attention_output(query_head, key_head, value_head):
        # Head dimension (used for scaling)
    d = key_head.shape[-1]

    # Create mask (prevent attending to future tokens)
    M = torch.triu(
        torch.full(
            (query_head.shape[0], key_head.shape[0]),
            float("-inf"),
            device=query_head.device,
            dtype=query_head.dtype,
        ),
        diagonal=1,
    )

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
        O_true[q_idx].unsqueeze(0),
        O_approx[q_idx].unsqueeze(0),
        dim=-1,
    )

    print("True attention on needle:", true_needle_attention.item())
    print("Approx attention on needle:", approx_needle_attention.item())
    print("Output cosine similarity:", cosine.item())

    return A_true, A_approx, O_true, O_approx


#------------------------EVALUATION--------------------------
# load model only once 
model,tokenizer = load_model()

# Choose the document containing the needle
path = f'document-haystack/AIG/AIG_10Pages/Text_TextNeedles/AIG_10Pages_TextNeedles_page_{1}.txt'

# Create the chat messages
messages, prompt, needle = get_messages(path, num_tokens=100)

# Add the true needle answer to the input, so we can inspect attention from the answer tokens
messages[1]["content"] += " " + needle

# Extract K, V, and Q
key_head, value_head, query_head = get_kvq(
    messages,
    layer_idx=0,
    head_idx=0,
    want_print=True,
    model=model,
    tokenizer=tokenizer,
)

# Find the exact token positions where the needle appears in the full model input
needle_positions = find_token_positions(tokenizer, messages, needle)
print("Needle:", needle)
print("Needle positions:", needle_positions)

# Approximate the key matrix using SVD method 1
key_approx = method_1(key_head)

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
plt.show()

