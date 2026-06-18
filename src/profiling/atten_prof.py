import os
from fagprojekt.model import (load_model, get_messages, get_kvq)
from fagprojekt.SVD import (method_1, method_2, method_3, method_4)
from fagprojekt.Hokus_pokus import (hokus_pokus, train, build_mlp)
from fagprojekt.SVD_Nystrom import nystrom_attention_approx

from pathlib import Path
import os
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.flop_counter import FlopCounterMode

prof_method_1 = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_method_2 = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_method_3 = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_method_4 = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_nystrom = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_hokus_pokus = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_flash_attention = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)
prof_naive_attention = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True)


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


def Nystrom_Attention(query_head, value_head, key_head, k, num_tokens, layer_idx, head_idx):
    prof_nystrom.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = nystrom_attention_approx(key_head, query_head, value_head, k=k)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_nystrom.step()  # Record the step for accurate timing
    prof_nystrom.stop()
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'Nystrom Attention FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/nystrom_attention_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_nystrom.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")



def Hokus_Pokus(query_head, value_head, key_head, method, layer_idx, head_idx, k, num_tokens, model, tokenizer):
    prof_hokus_pokus.start()
    with record_function("Train g_theta"):
        train_companies = sorted(['GoldmanSachs', 'HSBC', 'JPMorgan', 'Kroger', 'NewRiver', 'PNC', 'Reach', 'Sagicor', 'United', 'UPS', 'Vesuvius', 'WoltersKluwer'])
        test_companies  = sorted(['AIG', 'AmericanAirlines', 'APA'])

        train_paths = [f'{Path(f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles")}_page_{page}.txt' for company in train_companies for page in range(1,26)]
        test_paths = [f'{Path(f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles")}_page_{page}.txt' for company in test_companies for page in range(1,26)]

        method = "mlp"
        loss_method = 'mse'
        num_epochs = 30

        # Train model on the training paths (NOT inside FlopCounterMode: this runs
        # full model.generate() with grouped-query attention, which the installed
        # torch.utils.flop_counter's sdpa_flop_count can't handle and crashes on)
        g_theta = train(train_paths, method=method, layer_idx=layer_idx, head_idx=head_idx, k=k, model=model, tokenizer=tokenizer, loss_method=loss_method, tokens=num_tokens, plot_figure=True, num_epochs=num_epochs)

        # Save model
        model_path = f"models/g_theta_weights_{method}_k_{k}_epochs_{num_epochs}.pth"
        torch.save(g_theta.state_dict(), model_path)

        # load the g_theta model weights only once
        g_theta_loaded = build_mlp(k).to(next(g_theta.parameters()).device)
        g_theta_loaded.load_state_dict(torch.load(model_path, map_location=next(g_theta.parameters()).device))
        g_theta_loaded.eval()

        # Load saved model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = f"models/g_theta_weights_mlp_k_{k}_epochs_{num_epochs}.pth"
        g_theta = build_mlp(k).to(device)
        g_theta.load_state_dict(torch.load(model_path, map_location=device))
        g_theta.eval()

    with FlopCounterMode(display=False) as flop_counter:
        with record_function("Hokus Pokus attention"):
            result = hokus_pokus(query_head, value_head, key_head, k, method, g_theta)

    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_hokus_pokus.step()  # Record the step for accurate timing
    prof_hokus_pokus.stop()
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'Hokus Pokus FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/Hokus_Pokus_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(f"--- Iteration {i} ---\n")
        f.write(prof_hokus_pokus.key_averages().table(sort_by="self_cpu_time_total"))
        f.write("\n\n")
        



def flash_attention(query_head, value_head, key_head):
    prof_flash_attention.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = torch.nn.functional.scaled_dot_product_attention(query_head, key_head, value_head, is_causal=True)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof_flash_attention.step()  # Record the step for accurate timing
    prof_flash_attention.stop()
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
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
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
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

        SVD1(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k)
        SVD2(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k)
        SVD3(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k)
        SVD4(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k)

        Hokus_Pokus(query_head=query_head, value_head=value_head, key_head=key_head, method="mlp", layer_idx=layer_idx, head_idx=head_idx, k=k, num_tokens=num_tokens, model=model, tokenizer=tokenizer)
        Nystrom_Attention(query_head=query_head, value_head=value_head, key_head=key_head, k=k, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx)
        flash_attention(query_head=query_head, value_head=value_head, key_head=key_head)
        naive_attention(query_head=query_head, value_head=value_head, key_head=key_head)