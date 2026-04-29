# imports 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt

from fagprojekt.model import (load_model, get_kvq, get_messages,)
from fagprojekt.head_level_evel_utils import(find_token_positions, get_attention_output, evaluate_head, method_1_K, find_needle_heads)

#------------------------VARIABLES---------------------------


#------------------------EVALUATION--------------------------
# load model only once 
model,tokenizer = load_model()

# Choose the document containing the needle
path = f'document-haystack/AIG/AIG_10Pages/Text_TextNeedles/AIG_10Pages_TextNeedles_page_{6}.txt'

# Create the chat messages
messages, prompt, needle = get_messages(path, num_tokens=500)
print(messages)


# Find the exact token positions where the needle appears in the full model input
needle_positions = find_token_positions(tokenizer, messages, needle)
print("Needle:", needle)
print("Needle positions:", needle_positions)


# Find best needle head
needle_heads = find_needle_heads(
    model,
    tokenizer,
    messages,
    needle,
    top_k=20,
)

best = needle_heads[0]
layer_idx = best["layer"]
head_idx = best["head"]

print("Best needle head:")
print(best)


# Extract K, V, and Q
key_head, value_head, query_head = get_kvq(
    messages,
    layer_idx=layer_idx,
    head_idx=head_idx,
    want_print=False,
    model=model,
    tokenizer=tokenizer,
)

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



