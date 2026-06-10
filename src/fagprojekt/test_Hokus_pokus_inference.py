import torch

from fagprojekt.model import load_model
from fagprojekt.Hokus_pokus_inference import HokusPokusLlamaAttentionWrapper

"""
Lille test af Hokus Pokus-cache-wrapperen.

Vi loader modellen, wrapper ét attention-layer og ser om modellen stadig kan
generere et normalt svar, når det layer bruger vores egen Hokus Pokus-cache
i stedet for HuggingFace's normale KV-cache.

Hvis den ikke crasher og svaret giver mening, er insertionen basically okay.
"""


def reset_hokus_pokus_caches(model):
    """
    Nulstiller Hokus Pokus-caches i alle wrapped layers.

    Det er bare for at sikre, at en ny test ikke bruger cache fra en tidligere
    prompt eller generation.
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

    model.model.layers[layer_idx].self_attn = HokusPokusLlamaAttentionWrapper(
        original_attn=original_attn,
        rank=rank,
        g_theta=None,  # Bruger softmax(qB), fordi vi ikke har trænet MLP'en endnu.
    )

    reset_hokus_pokus_caches(model)

    messages = [
        {"role": "system", "content": "Answer briefly."},
        {
            "role": "user",
            "content": "The secret currency is euro. What is the secret currency?",
        },
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

    print("\n--- MODEL OUTPUT ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()