from sklearn.cluster import KMeans
import torch
from fagprojekt.model import get_messages, get_kvq, get_true_attention_values
from fagprojekt.SVD import compare_attention
from pathlib import Path
import mathfrom torch.profiler import profile, ProfilerActivity, record_function

prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],schedule = torch.profiler.schedule(wait=0,warmup=0,active=1),profile_memory=True, record_shapes=True, acc_events=True) 


def k_means_clustering(key_head, value_head, query_head, clusters=8):
    """ Perform K-means clustering on the key and value head representations and 
    compute attention values using the clustered representations. 
    
    Args:
        query_head: Query tensor of shape [T, C]
        value_head: Value tensor of shape [T, C]
        key_head: Key tensor of shape [T, C]
        clusters: The number of clusters to use in K-means clustering.
    Returns:
        attn_values: The attention values computed using the clustered key and value representations.
    """
    with record_function("k_means"):
        kmeans = KMeans(n_clusters=clusters)

        # concatenate key and value head for clustering
        kv_concat = torch.cat((key_head, value_head), dim=-1) # [T, 2C]

        # fit and transform to cluster space
        kmeans.fit(kv_concat.cpu().numpy())  # convert to CPU numpy array
        centroids = torch.tensor(kmeans.cluster_centers_, device=kv_concat.device, dtype=kv_concat.dtype)  # [clusters, 2C]

        # split the clustered space back into key and value components
        A,B = torch.chunk(centroids, 2, dim=-1) # [clusters, C], [clusters, C]
        
        # compute the attention values  using the clustered key and value representations
        d = query_head.shape[-1]
        attn_values = torch.softmax((query_head @ A.T) / math.sqrt(d), dim=-1) @ B
    return attn_values


def get_rmse_companies_K_means(model, tokenizer, layer_idx, head_idx, num_tokens, clusters, companies):
    rmse_errors=[]
    pages = range(1, 26)
    for company in companies:
        base_dir = Path(f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles")
        for page in pages:
            page_path = f'{base_dir}_page_{page}.txt'
            messages, _, _ = get_messages(page_path, num_tokens=num_tokens)
            key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False, model=model, tokenizer=tokenizer)

            true_attn = get_true_attention_values(query_head, key_head, value_head)
            attn_values_k_means = k_means_clustering(key_head, value_head, query_head, clusters=clusters)

            mse, _, _ = compare_attention(true_attn, attn_values_k_means, "K_means", want_print=False)
            rmse_errors.append(math.sqrt(mse))

    return rmse_errors




    

