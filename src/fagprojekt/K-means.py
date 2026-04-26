from sklearn.cluster import KMeans
import torch
from fagprojekt.model import get_messages, get_kvq


path = "document-haystack/AIG/AIG_5Pages/Text_TextNeedles/AIG_5Pages_TextNeedles_page_4.txt"
messages, _, _ = get_messages(path, num_tokens=100)

key_head, _, _ = get_kvq(messages, layer_idx=0, head_idx=0, want_print=True)

def k_means_clustering(query_head,value_head,key_head, clusters=8):
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

    kmeans = KMeans(n_clusters=clusters)

    # concatenate key and value head for clustering
    kv_concat = torch.cat((key_head, value_head), dim=-1) # [T, 2C]

    # fit and transform to cluster space
    kmeans.fit(kv_concat) 
    centroids = torch.tensor(kmeans.cluster_centers_, device=kv_concat.device, dtype=kv_concat.dtype)  # [clusters, 2C]

    # split the clustered space back into key and value components
    A,B = torch.chunk(centroids, 2, dim=-1) # [clusters, C], [clusters, C]
    
    # compute the attention values  using the clustered key and value representations
    attn_values = torch.softmax(query_head @ A.T, dim=-1) @ B

    return attn_values