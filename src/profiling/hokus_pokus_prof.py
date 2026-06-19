import os
from fagprojekt.model import load_model, get_messages, get_kvq
from fagprojekt.Hokus_pokus import hokus_pokus, build_mlp

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.flop_counter import FlopCounterMode

def load_g_theta(k):
    method = "mlp"
    num_epochs = 90

    # Load saved model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"models/g_theta_weights_{method}_k_{k}_epochs_{num_epochs}.pth"
    g_theta = build_mlp(k).to(device)
    g_theta.load_state_dict(torch.load(model_path, map_location=device))
    g_theta.eval()
    return g_theta


def Hokus_Pokus(query_head, value_head, key_head, method, layer_idx, head_idx, k, num_tokens, g_theta, i):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        with FlopCounterMode(display=False) as flop_counter:
            with record_function("Hokus Pokus attention"):
                result = hokus_pokus(query_head, value_head, key_head, k, method, g_theta)
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished before stopping the profiler
    with open(f"reports/figures/profiling/flops_hokus_pokus.txt", "a") as f:
        f.write(f'Hokus Pokus FLOP {i}:  {flop_counter.get_total_flops()} \n')
    with open(f"reports/figures/profiling/Hokus_Pokus_layer_{layer_idx}_head_{head_idx}_tokens_{num_tokens}_k_{k}.txt", "a") as f:
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

    model, tokenizer = load_model(want_print=False)

    # Load trained g_theta:
    g_theta = load_g_theta(k)

    for i in range(11):
        messages, _, _ = get_messages(f"document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_{i+1}.txt", num_tokens=num_tokens)
        key_head, value_head, query_head = get_kvq(model=model, tokenizer=tokenizer, messages=messages, layer_idx=layer_idx, head_idx=head_idx)

        Hokus_Pokus(query_head=query_head, value_head=value_head, key_head=key_head, method="mlp", layer_idx=layer_idx, head_idx=head_idx, k=k, num_tokens=num_tokens, g_theta=g_theta, i=i)

