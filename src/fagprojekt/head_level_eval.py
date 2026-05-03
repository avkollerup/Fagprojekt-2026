# imports 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt

from fagprojekt.model import (load_model, get_kvq, get_messages,)
from fagprojekt.head_level_eval_utils import(find_token_positions, evaluate_head, method_1_K, find_needle_heads)

#------------------------VARIABLES---------------------------
page_number = 6
num_tokens = 200
num_top_heads = 10
num_layers = None # write None if you want all layers
num_heads = None # write None if you want all heads

#------------------------EVALUATION--------------------------
def head_level_eval(path, num_tokens, file_num, model, tokenizer):

    # Create the chat messages
    messages, prompt, needle = get_messages(path, num_tokens=num_tokens)

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
        top_k=num_top_heads,
        num_layers=num_layers,
        num_heads=num_heads
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
    A_true, A_approx, O_true, O_approx, true_needle_attention, approx_needle_attention, cos_sim  = evaluate_head(
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

    plt.legend(loc="upper right")
    plt.title("Plot of Attention Values for Each Token")
    plt.xlabel("Token Number")
    plt.ylabel("Attention")
    plt.savefig(f"reports/figures/eval_{num_tokens}tokens_page_{file_num}.png", dpi=150)
    plt.close()

# load model only once 
model, tokenizer = load_model()
base_path = "document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_"

for i in range(25):
    file_path = f"{base_path}{i+1}.txt"
    head_level_eval(path=file_path, num_tokens=num_tokens, file_num=i+1, model=model, tokenizer=tokenizer)
