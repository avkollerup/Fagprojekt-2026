
import os
from fagprojekt.model import (load_model, get_messages, get_kvq)
from fagprojekt.SVD import (method_1)

from pathlib import Path
import os
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.flop_counter import FlopCounterMode

prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True, acc_events=True) 


def SVD1(key_head, query_head, value_head, num_tokens, layer_idx, head_idx, k):
    prof.start()
    with FlopCounterMode(display=False) as flop_counter:
        result = method_1(key_head, query_head, value_head, k=k)
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    prof.step()  # Record the step for accurate timing
    prof.stop()
    with open(f"reports/figures/profiling/method_1_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
    with open(f"reports/figures/profiling/flops.txt", "a") as f:
        f.write(f'Method 1 FLOP:  {flop_counter.get_total_flops()}')

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])
    k = 45 
    model, tokenizer = load_model(want_print=False)
    messages, _, _ = get_messages(f"document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_{1}.txt", num_tokens=num_tokens)
    key_head, value_head, query_head = get_kvq(model=model, tokenizer=tokenizer, messages=messages, layer_idx=layer_idx, head_idx=head_idx)

    SVD1(key_head=key_head, query_head=query_head, value_head=value_head, num_tokens=num_tokens, layer_idx=layer_idx, head_idx=head_idx, k=k)
        
