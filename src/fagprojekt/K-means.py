from sklearn.cluster import KMeans
from fagprojekt.model import get_messages, get_kvq


path = "document-haystack/AIG/AIG_5Pages/Text_TextNeedles/AIG_5Pages_TextNeedles_page_4.txt"
messages, _, _ = get_messages(path, num_tokens=100)

key_head, _, _ = get_kvq(messages, layer_idx=0, head_idx=0, want_print=True)

r = 8
kmeans = KMeans(n_clusters=r)
labels = kmeans.fit_predict(key_head)        
centroids = kmeans.cluster_centers_  
print(centroids)

B = centroids.T # ???