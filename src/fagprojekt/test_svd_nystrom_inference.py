import torch
import seaborn as sns
import numpy as np
from fagprojekt.model import (load_model, get_messages)
from fagprojekt.nystrom_unfilled import (patch_llama_attention, set_prefill, clear_nystrom,build_full_attention_matrix)
from fagprojekt.head_level_eval_utils import find_token_positions, get_random_messages
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity, record_function

prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True, acc_events=True) 

"""
Simple test of SVD-Nystrom attention.

We patch one Llama attention layer, run prefill once, then generate manually
one token at a time while the wrapper uses its compressed prompt cache.

If it does not crash, the insertion path basically works.
"""

def sample_next_token(logits):
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
# Sample fra multinomial distribution (sample fra top 50)


def main():
    print("running")
    model, tokenizer = load_model(want_print=True)
    model.eval()

    layer_idx = 5
    head_idx = 0
    lamba = 0.5
    local_window = 128

    patch_llama_attention(
        model=model,
        layers="all",
        rank=1,
        local_window=local_window,
        eps=1e-4,
        lamba=lamba,
        svd_mode="svd_k") 

    clear_nystrom(model)

    messages = [{"role": "system", "content": "Answer briefly."}, 
                {"role": "user", "content": "What is your name?",}]

    path = "document-haystack/BankOfMontreal/BankOfMontreal_25Pages/Text_TextNeedles/BankOfMontreal_25Pages_TextNeedles_page_9.txt"
    num_tokens = 400

    messages, _, needle = get_random_messages(path, num_tokens=num_tokens,local_window_size=local_window)

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
            with record_function("token generation"):
                next_token = sample_next_token(outputs.logits)
                generated.append(next_token)

            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_ids = torch.cat([input_ids] + generated, dim=-1)
    seq_len = input_ids.shape[-1]
    print("Length of input tokens:", seq_len)
    print("\n--- MODEL OUTPUT ---")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    needle_min = min(needle_pos)
    needle_max = max(needle_pos)
    needle_color = "darkgreen"

    print("\n--- ATTENTION MATRICES ---")
    for head in range(32):

        attn_matrix = build_full_attention_matrix(module=model.model.layers[layer_idx].self_attn, head_idx=head, generated_ids=generated_ids, softmax=True)
        value_range = torch.quantile(torch.abs(attn_matrix), 0.995).item()
        if value_range < 1e-4:
            value_range = 1e-4

        fig, ax_heat = plt.subplots(figsize=(12, 10))

        sns.heatmap( attn_matrix, cmap="vlag", vmin=-value_range, vmax=value_range, cbar_kws={"label": "attention value"}, ax=ax_heat)
        ax_heat.set_title(f"Attention Heatmap for Head {head} (Layer {layer_idx}, lambda={lamba})")
        ax_heat.set_xlabel("Key Position")
        ax_heat.set_ylabel("Query Position")

        ax_heat.axvline(seq_len, color="gray", linestyle="--", linewidth=1)
        ax_heat.axhline(seq_len, color="gray", linestyle="--", linewidth=1)

        ax_heat.plot([needle_min + 0.5, needle_max + 0.5], [1.02, 1.02], color=needle_color, linewidth=3, transform=ax_heat.get_xaxis_transform(), clip_on=False)
        ax_heat.text((needle_min + needle_max) / 2 + 0.5, 1.05, "needle", color=needle_color, ha="center", va="bottom", transform=ax_heat.get_xaxis_transform())

        ax_heat.plot([-0.08, -0.08], [needle_min + 0.5, needle_max + 0.5], color=needle_color, linewidth=3, transform=ax_heat.get_yaxis_transform(), clip_on=False)
        ax_heat.text(-0.08, (needle_min + needle_max) / 2 + 0.5, "needle", color=needle_color, ha="right", va="center", transform=ax_heat.get_yaxis_transform())


        plt.savefig(f"reports/figures/attention_heatmaps/k1_attention_heatmap_head{head}.png")
        plt.close()
     

if __name__ == "__main__":
    main()

