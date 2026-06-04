import torch

from fagprojekt.model import load_model
from fagprojekt.replaceable_llama_attention import SVDLlamaAttentionWrapper

"""
Lille test af SVD-cache-wrapperen.

Vi loader modellen, wrapper ét attention-layer og ser om modellen stadig kan
generere et normalt svar, når det layer bruger vores egen cache i stedet for
HuggingFace's cache.

Hvis den ikke crasher og svaret giver mening, er insertionen basically okay (tror jeg).
"""

def reset_svd_caches(model):
    """
    Nulstiller SVD-caches i alle wrapped layers.
    Just in case, så en ny test ikke bruger cache fra en tidligere test.
    """
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "reset_cache"):
            attn.reset_cache()


def main():
    model, tokenizer = load_model(want_print=True)
    model.eval()

    layer_idx = 5
    rank = 16

    original_attn = model.model.layers[layer_idx].self_attn

    model.model.layers[layer_idx].self_attn = SVDLlamaAttentionWrapper(
        original_attn=original_attn,
        rank=rank,
    )

    reset_svd_caches(model)

    messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "The secret currency is euro. What is the secret currency?"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()