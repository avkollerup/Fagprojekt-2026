import torch

from fagprojekt.model import load_model
from fagprojekt.nystrom_unfilled import (
    patch_llama_attention,
    set_prefill,
    clear_nystrom,
    SVDNystromLlamaAttention,
)


"""
Test of SVD-Nystrom / Hokus Pokus inference.

The test:
1. Loads Llama.
2. Replaces one attention layer.
3. Runs prefill once to build the compressed prompt cache.
4. Generates tokens one at a time.
5. Prints useful debug information during decoding.
"""


def print_wrapped_layer_debug(model):
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        if not isinstance(attn, SVDNystromLlamaAttention):
            continue

        print(f"\n--- WRAPPED LAYER {layer_idx} ---")
        print(f"prefill: {attn.prefill}")
        print(f"prefix_len: {attn.prefix_len}")
        print(f"generated_len: {attn.generated_len}")
        print(f"rank: {attn.rank}")
        print(f"lambda: {attn.lamba}")
        print(f"local_window: {attn.local_window}")

        if attn.prefix_k is not None:
            print(f"prefix_k shape: {tuple(attn.prefix_k.shape)}")
            print(f"prefix_v shape: {tuple(attn.prefix_v.shape)}")
            print(f"prefix_k norm: {attn.prefix_k.float().norm().item():.4f}")
            print(f"prefix_v norm: {attn.prefix_v.float().norm().item():.4f}")

        if attn.target_k is not None:
            print(f"target_k shape: {tuple(attn.target_k.shape)}")
            print(f"target_v shape: {tuple(attn.target_v.shape)}")
            print(f"target_k norm: {attn.target_k.float().norm().item():.4f}")
            print(f"target_v norm: {attn.target_v.float().norm().item():.4f}")

        if attn.B is not None:
            print(f"B shape: {tuple(attn.B.shape)}")
            print(f"B norm: {attn.B.float().norm().item():.4f}")
            print(f"B max abs: {attn.B.float().abs().max().item():.4f}")

        if attn.const_cache is not None:
            print(f"const_cache shape: {tuple(attn.const_cache.shape)}")
            print(f"const_cache norm: {attn.const_cache.float().norm().item():.4f}")
            print(f"const_cache max abs: {attn.const_cache.float().abs().max().item():.4f}")


def print_top_tokens(logits, tokenizer, top_k=5):
    next_logits = logits[:, -1, :]
    probs = torch.softmax(next_logits.float(), dim=-1)

    top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)

    print("\nTop next tokens:")
    for i in range(top_k):
        token_id = top_ids[0, i].item()
        prob = top_probs[0, i].item()
        token_text = tokenizer.decode([token_id])

        print(f"{i + 1}. id={token_id:6d} prob={prob:.4f} token={repr(token_text)}")


def choose_next_token(logits):
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def debug_decode(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens=20,
):
    generated_tokens = []

    with torch.no_grad():
        print("\n========== PREFILL ==========")

        set_prefill(model, True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        print(f"input_ids shape: {tuple(input_ids.shape)}")
        print(f"attention_mask shape: {tuple(attention_mask.shape)}")
        print(f"logits shape: {tuple(outputs.logits.shape)}")
        print(f"logits norm: {outputs.logits.float().norm().item():.4f}")
        print(f"logits max: {outputs.logits.float().max().item():.4f}")
        print(f"logits min: {outputs.logits.float().min().item():.4f}")

        print_top_tokens(outputs.logits, tokenizer, top_k=5)
        print_wrapped_layer_debug(model)

        next_token = choose_next_token(outputs.logits)
        generated_tokens.append(next_token)

        print("\nFirst generated token:")
        print(f"id: {next_token.item()}")
        print(f"text: {repr(tokenizer.decode(next_token[0]))}")

        print("\n========== DECODING ==========")

        set_prefill(model, False)

        for step in range(max_new_tokens):
            outputs = model(
                input_ids=next_token,
                attention_mask=torch.ones_like(next_token),
                use_cache=False,
            )

            print(f"\n--- Decode step {step + 1} ---")
            print(f"input id: {next_token.item()}")
            print(f"input text: {repr(tokenizer.decode(next_token[0]))}")
            print(f"logits norm: {outputs.logits.float().norm().item():.4f}")
            print(f"logits max: {outputs.logits.float().max().item():.4f}")
            print(f"logits min: {outputs.logits.float().min().item():.4f}")

            print_top_tokens(outputs.logits, tokenizer, top_k=5)
            print_wrapped_layer_debug(model)

            next_token = choose_next_token(outputs.logits)
            generated_tokens.append(next_token)

            print(f"chosen next id: {next_token.item()}")
            print(f"chosen next text: {repr(tokenizer.decode(next_token[0]))}")

            if next_token.item() == tokenizer.eos_token_id:
                print("EOS token reached.")
                break

    output_ids = torch.cat([input_ids] + generated_tokens, dim=-1)

    print("\n========== FINAL OUTPUT ==========")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return output_ids


def main():
    model, tokenizer = load_model(want_print=True)
    model.eval()

    layer_idx = 5

    patch_llama_attention(
        model=model,
        layers=layer_idx,
        rank=16,
        local_window=128,
        eps=1e-4,
        lamba=0.0,  # Start with 0.0 to test local path first.
        svd_mode="svd_k",
    )

    clear_nystrom(model)

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

    debug_decode(
        model=model,
        tokenizer=tokenizer,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,
    )


if __name__ == "__main__":
    main()