import os
from fagprojekt.model import load_model, get_messages, get_kvq
from fagprojekt.SVD_Nystrom import nystrom_attention_approx

import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode

prof_nystrom = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_flash_attention = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_naive_attention = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)


def Nystrom_Attention(query_head, value_head, key_head, k, num_tokens, layer_idx, head_idx):
    prof_nystrom.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = nystrom_attention_approx(key_head, query_head, value_head, k=k)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_nystrom.step()  # Record the step for accurate timing
    prof_nystrom.stop()
    with open(f"reports/figures/profiling/flops_svd_nystrom.txt", "a") as f:
        f.write(f'Nystrom Attention FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/nystrom_attention_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_nystrom.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")


def flash_attention(query_head, value_head, key_head):
    prof_flash_attention.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = torch.nn.functional.scaled_dot_product_attention(query_head, key_head, value_head, is_causal=True)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_flash_attention.step()  # Record the step for accurate timing
    prof_flash_attention.stop()
    with open(f"reports/figures/profiling/flops_svd_nystrom.txt", "a") as f:
        f.write(f'Flash Attention FLOP {i} :  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/flash_attention_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_flash_attention.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def naive_attention(query_head, value_head, key_head):
    prof_naive_attention.start()
    with FlopCounterMode(display = False) as flop_counter:
        T = query_head.shape[-2]
        M = torch.triu(torch.full((T,T), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)
        attn_scores = torch.matmul(query_head, key_head.transpose(-2, -1)) / (query_head.size(-1) ** 0.5)
        attn_weights = torch.nn.functional.softmax(M + attn_scores, dim=-1)
        result = torch.matmul(attn_weights, value_head)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_naive_attention.step()  # Record the step for accurate timing
    prof_naive_attention.stop()
    with open(f"reports/figures/profiling/flops_svd_nystrom.txt", "a") as f:
        f.write(f'Naive Attention FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/naive_attention_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_naive_attention.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

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

        Nystrom_Attention(query_head=query_head, value_head=value_head, key_head=key_head, k=k, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx)

        flash_attention(query_head=query_head, value_head=value_head, key_head=key_head)
        naive_attention(query_head=query_head, value_head=value_head, key_head=key_head)
