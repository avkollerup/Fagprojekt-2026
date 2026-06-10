import torch
import os

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
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "reset_cache"):
            attn.reset_cache()


def hokus_pokus_generate(model, tokenizer, inputs, max_new_tokens=20):
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    reset_hokus_pokus_caches(model)

    # --- PREFILL ---
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated  = [next_token.item()]

    # --- DECODING ---
    for _ in range(max_new_tokens - 1):
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(1, 1, device=model.device, dtype=attention_mask.dtype)
        ], dim=1)

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                use_cache=False,
            )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

        if next_token.item() == tokenizer.eos_token_id:
            break

    all_ids = torch.cat([input_ids[0], torch.tensor(generated, device=model.device)])
    return tokenizer.decode(all_ids, skip_special_tokens=True)


def main():
    model, tokenizer = load_model(want_print=True)
    model.eval()

    layer_idx     = 5
    rank          = 64
    original_attn = model.model.layers[layer_idx].self_attn

    model.model.layers[layer_idx].self_attn = HokusPokusLlamaAttentionWrapper(
        original_attn=original_attn,
        rank=rank,
        g_theta=None,
    )

    messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user",   "content": "The secret currency is euro. What is the secret currency?"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    result = hokus_pokus_generate(model, tokenizer, inputs, max_new_tokens=20)

    # Print cache norms after prefill — hp_prompt_cache is populated now
    print("\n--- CACHE DIAGNOSTICS (layer 5) ---")
    for kv_head_idx, head_cache in model.model.layers[layer_idx].self_attn.hp_prompt_cache.items():
        print(f"  head {kv_head_idx}: cache norm = {head_cache['cache'].norm():.3f}, "
              f"B norm = {head_cache['B'].norm():.3f}")
        if kv_head_idx >= 3:
            print("  ...")
            break

    print("\n--- MODEL OUTPUT ---")
    print(result)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    main()