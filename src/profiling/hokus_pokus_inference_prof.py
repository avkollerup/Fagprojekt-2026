import torch
import os
from torch.profiler import profile, record_function, ProfilerActivity
from fagprojekt.model import load_model, get_messages
from fagprojekt.nystrom_unfilled import (patch_llama_attention, set_prefill, build_full_attention_matrix)
from fagprojekt.test_hokus_pokus_inference import clear_hokuspokus, build_mlp, sample_next_token


# Some GPU node allocations on the cluster come up with no CUDA device visible;
# fall back to CPU-only profiling instead of crashing on torch.cuda.synchronize().
_CUDA = torch.cuda.is_available()
_ACTIVITIES = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if _CUDA else [])


def _cuda_sync():
    if _CUDA:
        torch.cuda.synchronize()


prof_hokus = profile(activities=_ACTIVITIES, profile_memory=True, record_shapes=True)

def hokuspokus_inference(messages, model, tokenizer, layer_idx, lamba, local_window):
    prof_hokus.start()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # load g_theta
    with record_function("Loading g_theta"):
        model_path = "models/g_theta_weights_mlp_k_45_epochs_90.pth"
        g_theta = build_mlp(45).to(device)
        g_theta.load_state_dict(torch.load(model_path, map_location=device))
        g_theta.eval()

    with record_function("Patching LLaMA attention with Hokus Pokus"):
        patch_llama_attention(
            model=model,
            layers="all",
            rank=45,
            local_window=local_window,
            eps=1e-4,
            lamba=lamba,
            svd_mode="svd_k",
            loaded_g_theta=g_theta)

    clear_hokuspokus(model)


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
        with record_function("Hokus Pokus prefill"):
            set_prefill(model, True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # First generated token.
        with record_function("Hokus Pokus first token generation"):
            next_token = sample_next_token(outputs.logits)
            generated = [next_token]

        # Decoding: use compressed prompt cache + local attention.
        with record_function("Hokus Pokus decoding"):
            set_prefill(model, False)

        for _ in range(20):
            outputs = model(input_ids=next_token, attention_mask=torch.ones_like(next_token), use_cache=False)
            with record_function("token generation"):
                next_token = sample_next_token(outputs.logits)
                generated.append(next_token)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_ids = torch.cat([input_ids] + generated, dim=-1)
    with record_function("Hokus Pokus attention for all heads"):
        for head in range(32):

            attn_matrix = build_full_attention_matrix(module=model.model.layers[layer_idx].self_attn, head_idx=head, generated_ids=generated_ids, softmax=True)
            value_range = torch.quantile(torch.abs(attn_matrix), 0.995).item()
            if value_range < 1e-4:
                value_range = 1e-4
    _cuda_sync()
    prof_hokus.step()
    prof_hokus.stop()
    with open("reports/figures/profiling/hokuspokus_inference_metrics.txt", "a") as f:
        f.write(prof_hokus.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])

    model, tokenizer = load_model(want_print=False)

    for i in range(11):
        messages, _, _ = get_messages(f"document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_{i+1}.txt", num_tokens=num_tokens)

        hokuspokus_inference(messages=messages, model=model, tokenizer=tokenizer, layer_idx=layer_idx, lamba=0.5, local_window=128)
