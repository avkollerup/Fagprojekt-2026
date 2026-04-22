import torch
import numpy as np

# # load hub token
# from dotenv import load_dotenv
# load_dotenv()

# from huggingface_hub import login
# import os
# login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])


# We have used three metrics: 
# - MSE for raw error (might look small because of small values)
# - Frobenius norm for scale-independent accuracy
# - Cosine similarity to capture structural (attention pattern) similarity
def compare_attention(true_attn, approx_attn, name):
    mse = torch.mean((true_attn - approx_attn) ** 2).item()
    rel_frob = (torch.norm(true_attn - approx_attn, p="fro") / torch.norm(true_attn, p="fro")).item()
    cos = torch.nn.functional.cosine_similarity(true_attn.flatten(), approx_attn.flatten(), dim=0).item()

    print(f"{name}:")
    print(f"  MSE: {mse:.6e}")
    print(f"  Relative Frobenius error: {rel_frob:.6e}")
    print(f"  Cosine similarity: {cos:.6f}\n")


# do big principal component analysis
def pca_analysis():
    # imports 
    from fagprojekt.SVD import method_1,method_2,method_3
    from fagprojekt.model import (
    load_model,
    get_kvq,
    get_messages,
    get_true_attention_values
    )

    from collections import defaultdict
    import matplotlib.pyplot as plt

    # load model only once
    model,tokenizer = load_model()

    # performance lists
    method_1_dict = defaultdict(list)
    method_2_dict = defaultdict(list)
    method_3_dict = defaultdict(list)
    
    # iterate over pages ( not right now )
    for page in range(1,2):
        # get respone, kv cache and attention values
        path = f'document-haystack/AIG/AIG_10Pages/Text_TextNeedles/AIG_10Pages_TextNeedles_page_{page}.txt'
        messages, prompt, needle = get_messages(path, num_tokens=100)

        # perform for each head
        heads = range (0,1)
        for head in heads:
            key_head, value_head, query_head = get_kvq(messages, layer_idx=0, head_idx=head, want_print=True, model=model, tokenizer=tokenizer)
            true_attn_values = get_true_attention_values(query_head, key_head, value_head)

            # test different number of components
            components_list = list( map( int, np.linspace(10,200,10)))
            for num_components in components_list:
                attn_values_method_1 = method_1(key_head, query_head, value_head, k=num_components)
                attn_values_method_2 = method_2(key_head, query_head, value_head, k=num_components)
                attn_values_method_3 = method_3(key_head, query_head, value_head, k=num_components)

                true_attn_values = get_true_attention_values(query_head, key_head, value_head)

                # save to the lists
                method_1_dict[head].append(torch.mean((true_attn_values - attn_values_method_1)**2).item())
                method_2_dict[head].append(torch.mean((true_attn_values - attn_values_method_2)**2).item())
                method_3_dict[head].append(torch.mean((true_attn_values - attn_values_method_3)**2).item())
    
    # plot the analysis
    num_heads = len(method_1_dict)
    fig, axes = plt.subplots(nrows=1, ncols=num_heads, squeeze=False)
    axes = axes.flatten()
    for i, head in enumerate(method_1_dict.keys()):
        ax = axes[i]

        ax.plot(components_list, method_1_dict[head], label="K")
        ax.plot(components_list, method_2_dict[head], label="V")
        ax.plot(components_list, method_3_dict[head], label="K & V")

        ax.set_title(f'Head {head}')
        ax.set_ylabel('MSE')
        ax.set_xlabel('Num. components (k)')
        ax.legend()
    fig.suptitle('Principent component analysis over heads')
    plt.show()

            
if __name__ == "__main__":
    pca_analysis()