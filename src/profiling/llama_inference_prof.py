import torch
import os
from torch.profiler import profile, record_function, ProfilerActivity
from fagprojekt.llama import compute_attention_weights, extract_full_kv
from fagprojekt.model import extract_query, load_model, get_messages


# Some GPU node allocations on the cluster come up with no CUDA device visible;
# fall back to CPU-only profiling instead of crashing on torch.cuda.synchronize().
_CUDA = torch.cuda.is_available()
_ACTIVITIES = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if _CUDA else [])


def _cuda_sync():
    if _CUDA:
        torch.cuda.synchronize()


prof_llama = profile(activities=_ACTIVITIES, profile_memory=True, record_shapes=True)

def llama_attention(messages, model, tokenizer, layer_idx = 5, head_idx = 0):
    # This function can be used to profile the original LLaMA attention for comparison.
    prof_llama.start()
    model.eval()

    with record_function("tokenization"):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

    # Generate with KV cache - NO output_attentions to save memory
    with record_function("LLama output generation"):
        if _CUDA:
            torch.cuda.empty_cache()
        with torch.no_grad():
            generation_outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # Reduced from 21
                return_dict_in_generate=True,
                use_cache=True,
                output_attentions=False,
                output_scores=False
            )
        generated_ids = generation_outputs.sequences[0]
        query_inputs = {
            "input_ids": generated_ids.unsqueeze(0),
            "attention_mask": torch.ones_like(generated_ids).unsqueeze(0),
        }

    # extract_query/extract_full_kv each run their own full model(...) forward
    with record_function("QKV extraction"):
        query, query_head = extract_query(model, query_inputs, layer_idx, head_idx)
        _, key_head, _, value_head = extract_full_kv(model, query_inputs, layer_idx, head_idx)

    # Convert to float32 for numerical stability and move to CPU
    query_head = query_head.to(torch.float32).cpu()
    key_head = key_head.to(torch.float32).cpu()

    head_dim = query_head.shape[-1]

    with record_function("LLama attention compute"):
        full_attn = compute_attention_weights(query_head, key_head, head_dim=head_dim)

    _cuda_sync()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_llama.step()  # Record the step for accurate timing
    prof_llama.stop()
    with open(f"reports/figures/profiling/llama_attention_inference.txt", "a") as f:
        f.write(prof_llama.key_averages().table(sort_by="self_cpu_time_total"))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])

    model, tokenizer = load_model(want_print=False)

    for i in range(11):
        messages, _, _ = get_messages(f"document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_{i+1}.txt", num_tokens=num_tokens)

        llama_attention(messages=messages, model=model, tokenizer=tokenizer, layer_idx=layer_idx, head_idx=head_idx)
