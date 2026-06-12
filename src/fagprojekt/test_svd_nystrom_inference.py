import torch

from fagprojekt.model import (load_model, get_messages)
from fagprojekt.nystrom_unfilled import (patch_llama_attention, set_prefill, clear_nystrom)


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

    patch_llama_attention(
        model=model,
        layers="all",
        rank=60,
        local_window=128,
        eps=1e-4,
        lamba=0.01,
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

    print("\n--- MODEL OUTPUT ---")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()