import torch
import os
from torch.profiler import profile, record_function, ProfilerActivity
from fagprojekt.llama import compute_attention_weights, extract_full_kv
from fagprojekt.model import (extract_query, get_kvq, load_model, get_messages)
from fagprojekt.nystrom_unfilled import (patch_llama_attention, set_prefill, clear_nystrom,build_full_attention_matrix)
from fagprojekt.test_svd_nystrom_inference import sample_next_token
from torch.utils.flop_counter import FlopCounterMode


prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True)

def nystrom_inference(messages, layer_idx = 5, lamba = 0.5, local_window = 128):
    prof.start()
    model, tokenizer = load_model(want_print=True)
    model.eval()
    with FlopCounterMode(display=False) as flop_counter:
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

    torch.cuda.synchronize()
    prof.step()
    prof.stop()
    with open(f'reports/figures/profiling/nystrom_inference_metrics.txt', "a") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'Nystrom Inference FLOP {i}:  {flop_counter.get_total_flops()} \n')


def llama_attention(messages, layer_idx = 5, head_idx = 0):
    # This function can be used to profile the original LLaMA attention for comparison.
    prof.start()
    model.eval()
    with FlopCounterMode(display=False) as flop_counter:
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
            query_inputs = {
            "input_ids": generated_ids,
            "attention_mask": torch.ones_like(generated_ids),
        }
        query, query_head = extract_query(model, query_inputs, layer_idx, head_idx)

        # Extract full keys/values for the same full generated sequence so generated tokens are covered
        _, key_head, _, value_head = extract_full_kv(model, query_inputs, layer_idx, head_idx)

        # Convert to float32 for numerical stability and move to CPU
        query_head = query_head.to(torch.float32).cpu()
        key_head = key_head.to(torch.float32).cpu()
        
        head_dim = query_head.shape[-1]
        with record_function("LLama attention compute"):

            full_attn = compute_attention_weights(query_head, key_head, head_dim=head_dim)
            
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof.step()  # Record the step for accurate timing
    prof.stop()
    with open(f"reports/figures/profiling/llama_attention.txt", "a") as f:
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'LLama Attention FLOP {i}:  {flop_counter.get_total_flops()} \n')







    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])
    k = 45
    
    model, tokenizer = load_model(want_print=False)

    for i in range(11):
        messages, _, _ = get_messages(f"document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_{i+1}.txt", num_tokens=num_tokens)
        key_head, value_head, query_head = get_kvq(model=model, tokenizer=tokenizer, messages=messages, layer_idx=layer_idx, head_idx=head_idx)

        nystrom_inference(messages =messages)
        llama_attention(messages=messages, layer_idx=layer_idx, head_idx=head_idx)

