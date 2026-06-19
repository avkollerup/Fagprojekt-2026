import os
from fagprojekt.model import load_model, get_messages, get_kvq
from fagprojekt.SVD import method_1, method_2, method_3, method_4
from fagprojekt.K_means import k_means_clustering
from fagprojekt.SVD_Nystrom import nystrom_attention_approx

import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode


def SVD1(key_head, query_head, value_head, num_tokens, layer_idx, head_idx, k, i):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with FlopCounterMode(display=False) as flop_counter:
            result = method_1(key_head, query_head, value_head, k=k)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    with open(f"reports/figures/profiling/flops_svd_nystrom_kmeans.txt", "a") as f:
        f.write(f'Method 1 FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/method_1_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def SVD2(key_head, query_head, value_head, num_tokens, layer_idx, head_idx, k, i):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with FlopCounterMode(display=False) as flop_counter:
            result = method_2(key_head, query_head, value_head, k=k)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    with open(f"reports/figures/profiling/flops_svd_nystrom_kmeans.txt", "a") as f:
        f.write(f'Method 2 FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/method_2_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def SVD3(key_head, query_head, value_head, num_tokens, layer_idx, head_idx, k, i):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with FlopCounterMode(display=False) as flop_counter:
            result = method_3(key_head, query_head, value_head, k=k)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    with open(f"reports/figures/profiling/flops_svd_nystrom_kmeans.txt", "a") as f:
        f.write(f'Method 3 FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/method_3_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def SVD4(key_head, query_head, value_head, num_tokens, layer_idx, head_idx, k, i):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with FlopCounterMode(display=False) as flop_counter:
            result = method_4(key_head, query_head, value_head, k=k)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    with open(f"reports/figures/profiling/flops_svd_nystrom_kmeans.txt", "a") as f:
        f.write(f'Method 4 FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/method_4_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def Nystrom_Attention(query_head, value_head, key_head, k, num_tokens, layer_idx, head_idx, i):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with FlopCounterMode(display=False) as flop_counter:
            result = nystrom_attention_approx(key_head, query_head, value_head, k=k)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    with open(f"reports/figures/profiling/flops_svd_nystrom_kmeans.txt", "a") as f:
        f.write(f'Nystrom Attention FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/nystrom_attention_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def flash_attention(query_head, value_head, key_head, num_tokens, layer_idx, head_idx, i):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with FlopCounterMode(display=False) as flop_counter:
            result = torch.nn.functional.scaled_dot_product_attention(query_head, key_head, value_head, is_causal=True)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    with open(f"reports/figures/profiling/flops_svd_nystrom_kmeans.txt", "a") as f:
        f.write(f'Flash Attention FLOP {i} :  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/flash_attention_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def naive_attention(query_head, value_head, key_head, num_tokens, layer_idx, head_idx, i):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with FlopCounterMode(display = False) as flop_counter:
            T = query_head.shape[-2]
            M = torch.triu(torch.full((T,T), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)
            attn_scores = torch.matmul(query_head, key_head.transpose(-2, -1)) / (query_head.size(-1) ** 0.5)
            attn_weights = torch.nn.functional.softmax(M + attn_scores, dim=-1)
            result = torch.matmul(attn_weights, value_head)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    with open(f"reports/figures/profiling/flops_svd_nystrom_kmeans.txt", "a") as f:
        f.write(f'Naive Attention FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/naive_attention_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def KMeans_Attention(query_head, value_head, key_head, num_tokens, layer_idx, head_idx, clusters, i):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with FlopCounterMode(display=False) as flop_counter:
            result = k_means_clustering(key_head, value_head, query_head, clusters=clusters)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    with open(f"reports/figures/profiling/flops_svd_nystrom_kmeans.txt", "a") as f:
        f.write(f'K-means FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/kmeans_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_clusters_{clusters}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])
    k = 45
    clusters = 45

    model, tokenizer = load_model(want_print=False)

    for i in range(11):
        messages, _, _ = get_messages(f"document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_{i+1}.txt", num_tokens=num_tokens)
        key_head, value_head, query_head = get_kvq(model=model, tokenizer=tokenizer, messages=messages, layer_idx=layer_idx, head_idx=head_idx)

        SVD1(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k, i=i)
        SVD2(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k, i=i)
        SVD3(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k, i=i)
        SVD4(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k, i=i)

        Nystrom_Attention(query_head=query_head, value_head=value_head, key_head=key_head, k=k, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, i=i)

        flash_attention(query_head=query_head, value_head=value_head, key_head=key_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, i=i)
        naive_attention(query_head=query_head, value_head=value_head, key_head=key_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, i=i)

        KMeans_Attention(query_head=query_head, value_head=value_head, key_head=key_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, clusters=clusters, i=i)
