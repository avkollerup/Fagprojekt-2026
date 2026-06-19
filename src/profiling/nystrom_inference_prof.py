import torch
import os
from torch.profiler import profile, record_function, ProfilerActivity
from fagprojekt.model import load_model, get_messages
from fagprojekt.nystrom_unfilled import (patch_llama_attention, set_prefill, clear_nystrom, build_full_attention_matrix)
from fagprojekt.test_svd_nystrom_inference import sample_next_token


# Some GPU node allocations on the cluster come up with no CUDA device visible;
# fall back to CPU-only profiling instead of crashing on torch.cuda.synchronize().
_CUDA = torch.cuda.is_available()
_ACTIVITIES = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if _CUDA else [])


def _cuda_sync():
    if _CUDA:
        torch.cuda.synchronize()


prof_nystrom = profile(activities=_ACTIVITIES, profile_memory=True, record_shapes=True)

def nystrom_inference(messages, model, tokenizer, layer_idx = 5, lamba = 0.5, local_window = 128):
    prof_nystrom.start()
    model.eval()

    with record_function("Nystrom attention patching"):
        patch_llama_attention(
            model=model,
            layers="all",
            rank=60,
            local_window=local_window,
            eps=1e-4,
            lamba=lamba,
            svd_mode="svd_k")

        clear_nystrom(model)


    with record_function("tokenization"):
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
        with record_function("first token generation"):
            next_token = sample_next_token(outputs.logits)
            generated = [next_token]

        # Decoding: use compressed prompt cache + local attention.
        set_prefill(model, False)
        with record_function("total token generation"):
            for _ in range(20):
                outputs = model(input_ids=next_token, attention_mask=torch.ones_like(next_token), use_cache=False)
                with record_function("single token generation"):
                    next_token = sample_next_token(outputs.logits)
                    generated.append(next_token)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        generated_ids = torch.cat([input_ids] + generated, dim=-1)

        with record_function("Attention for all heads"):
            for head in range(32):

                attn_matrix = build_full_attention_matrix(module=model.model.layers[layer_idx].self_attn, head_idx=head, generated_ids=generated_ids, softmax=True)
                value_range = torch.quantile(torch.abs(attn_matrix), 0.995).item()
                if value_range < 1e-4:
                    value_range = 1e-4

    _cuda_sync()
    prof_nystrom.step()
    prof_nystrom.stop()
    with open(f'reports/figures/profiling/nystrom_inference_metrics.txt', "a") as f:
        f.write(prof_nystrom.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])

    model, tokenizer = load_model(want_print=False)

    for i in range(11):
        messages, _, _ = get_messages(f"document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_{i+1}.txt", num_tokens=num_tokens)

        nystrom_inference(messages=messages, model=model, tokenizer=tokenizer, layer_idx=layer_idx)
