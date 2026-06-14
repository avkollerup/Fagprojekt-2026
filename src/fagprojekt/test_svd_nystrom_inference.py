import torch
import seaborn as sns
import numpy as np
from fagprojekt.model import (load_model, get_messages)
from fagprojekt.nystrom_unfilled import (patch_llama_attention, set_prefill, clear_nystrom)
import matplotlib.pyplot as plt

"""
Simple test of SVD-Nystrom attention.

We patch one Llama attention layer, run prefill once, then generate manually
one token at a time while the wrapper uses its compressed prompt cache.

If it does not crash, the insertion path basically works.
"""

def sample_next_token(logits):
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def main():
    print("running")
    model, tokenizer = load_model(want_print=True)
    model.eval()

    layer_idx = 5
    head_idx = 0
    lamba = 0.01

    patch_llama_attention(
        model=model,
        layers="all",
        rank=60,
        local_window=128,
        eps=1e-4,
        lamba=lamba,
        svd_mode="svd_k")

    clear_nystrom(model)

    # messages = [{"role": "system", "content": "Answer briefly."}, 
    #             {"role": "user", "content": "What is your name?",}]

    path = "document-haystack/BankOfMontreal/BankOfMontreal_25Pages/Text_TextNeedles/BankOfMontreal_25Pages_TextNeedles_page_9.txt"
    num_tokens = 400

    messages, _, _ = get_messages(path, num_tokens=num_tokens)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
        ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        # Prefill: build compressed prompt cache.
        set_prefill(model, True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # First generated token.
        next_token = sample_next_token(outputs.logits)
        generated = [next_token]

        # Decoding: use compressed prompt cache + local attention.
        set_prefill(model, False)

        for _ in range(20):
            outputs = model(input_ids=next_token, attention_mask=torch.ones_like(next_token), use_cache=False)

            next_token = sample_next_token(outputs.logits)
            generated.append(next_token)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_ids = torch.cat([input_ids] + generated, dim=-1)

    seq_len = input_ids.shape[-1]
    print("Length of input tokens:", seq_len)
    print("\n--- MODEL OUTPUT ---")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    print("\n--- ATTENTION MATRICES ---")

    rows = []

    attn_module = model.model.layers[layer_idx].self_attn
    # prefill scores: [batch, heads, seq_len, seq_len]
    prefill_ = attn_module.prefix_attention_scores[0, head_idx]
    print(f"Prefill attention score matrix shape: {prefill_.shape}")

    for t, (local_scores, global_scores, local_pos) in enumerate(attn_module.attention_matrices):

        total_len = seq_len + t + 1
        row = torch.full((1,seq_len+len(generated_ids)), float(0)) # initialize with zeros
        print(f"Token {t}: local_pos={local_pos}, local_scores shape={local_scores.shape}, global_scores shape={global_scores.shape}")
        global_len = global_scores.shape[-1]
        # Global (prompt) attends only over prompt keys, not the full seq_len
        row[:, :global_len] = global_scores[0, head_idx, 0, :]

        # Local
        row[0,local_pos] = local_scores[0, head_idx, 0, :].view(1,-1) * (1 - lamba) + row[0,local_pos] * lamba 

        rows.append(row)

    # Stack into matrix
    attn_matrix = torch.stack(rows, dim=1) 

    # prefill_matrix = model.model.layers[layer_idx].self_attn.prefix_attention_scores
    # prefill_matrix = prefill_matrix[0, head_idx]  # Take the first batch and specified head
    
    # big_attention_matrix = np.zeros((seq_len + len(generated), seq_len + len(generated)), dtype=np.float32)
    # # insert the prefill block using raw scores before softmax
    # #big_attention_matrix[:prefill_matrix.shape[0], :prefill_matrix.shape[1]] = prefill_matrix.cpu().to(torch.float32).numpy()

    # # insert the global and local attention matrices for generated tokens
    # attn_module = model.model.layers[layer_idx].self_attn
    # for i in range(len(attn_module.attention_matrices)):
    #     local_attn, global_attn = attn_module.attention_matrices[i]
    #     global_slice = global_attn[0, head_idx].cpu().to(torch.float32).numpy()
    #     local_slice = local_attn[0, head_idx].cpu().to(torch.float32).numpy()
    #     row_index = seq_len + i
    #     # global attention covers prompt tokens only
    #     big_attention_matrix[row_index, : global_slice.shape[-1]] = global_slice
    #     # local attention covers prompt + generated keys in order, so align at column 0
    #     big_attention_matrix[row_index, : local_slice.shape[-1]] += local_slice.squeeze() * (1 - lamba)

    #print(big_attention_matrix.max())
    sns.heatmap(attn_matrix, cmap='vlag', cbar=True)
    plt.savefig("reports/figures/attention_heatmap.png")
    print("Saved heatmap")

        

if __name__ == "__main__":
    main()