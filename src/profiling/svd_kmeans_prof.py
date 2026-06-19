import os
from fagprojekt.model import load_model, get_messages, get_kvq
from fagprojekt.SVD import method_1, method_2, method_3, method_4
from fagprojekt.K_means import k_means_clustering

import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode

prof_method_1 = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_method_2 = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_method_3 = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_method_4 = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_kmeans = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)


def SVD1(key_head, query_head, value_head, num_tokens, layer_idx, head_idx, k):
    prof_method_1.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = method_1(key_head, query_head, value_head, k=k)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_method_1.step()  # Record the step for accurate timing
    prof_method_1.stop()
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'Method 1 FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/method_1_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_method_1.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def SVD2(key_head, query_head, value_head, num_tokens, layer_idx, head_idx, k):
    prof_method_2.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = method_2(key_head, query_head, value_head, k=k)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_method_2.step()  # Record the step for accurate timing
    prof_method_2.stop()
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'Method 2 FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/method_2_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_method_2.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def SVD3(key_head, query_head, value_head, num_tokens, layer_idx, head_idx, k):
    prof_method_3.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = method_3(key_head, query_head, value_head, k=k)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_method_3.step()  # Record the step for accurate timing
    prof_method_3.stop()
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'Method 3 FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/method_3_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_method_3.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")

def SVD4(key_head, query_head, value_head, num_tokens, layer_idx, head_idx, k):
    prof_method_4.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = method_4(key_head, query_head, value_head, k=k)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_method_4.step()  # Record the step for accurate timing
    prof_method_4.stop()
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'Method 4 FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/method_4_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_method_4.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")


def KMeans_Attention(query_head, value_head, key_head, num_tokens, layer_idx, head_idx, clusters):
    prof_kmeans.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = k_means_clustering(key_head, value_head, query_head, clusters=clusters)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_kmeans.step()  # Record the step for accurate timing
    prof_kmeans.stop()
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'K-means FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/kmeans_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_clusters_{clusters}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_kmeans.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])
    k = 45
    clusters = 8

    model, tokenizer = load_model(want_print=False)

    for i in range(11):
        messages, _, _ = get_messages(f"document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_{i+1}.txt", num_tokens=num_tokens)
        key_head, value_head, query_head = get_kvq(model=model, tokenizer=tokenizer, messages=messages, layer_idx=layer_idx, head_idx=head_idx)

        SVD1(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k)
        SVD2(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k)
        SVD3(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k)
        SVD4(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k)
        KMeans_Attention(query_head=query_head, value_head=value_head, key_head=key_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, clusters=clusters)
