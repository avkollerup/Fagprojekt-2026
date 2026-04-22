# imports 
from fagprojekt.SVD import method_1,method_2,method_3
from fagprojekt.model import (
load_model,
get_kvq,
get_messages,
get_true_attention_values,
extract_KV,
extract_query
)

from collections import defaultdict
import matplotlib.pyplot as plt

# load model only once
model,tokenizer = load_model()

# get respone, kv cache and attention values
path = f'document-haystack/AIG/AIG_10Pages/Text_TextNeedles/AIG_10Pages_TextNeedles_page_{1}.txt'
messages, prompt, needle = get_messages(path, num_tokens=100)

messages[1]["content"] += " " + needle

key_head, value_head, query_head = get_kvq(messages)

print(key_head, value_head, query_head)

# Write down token locations of needle
needle_location = []

# Calculate approximation of K and V matrices (SVD)
K_approx = 0
V_approx = 0

# Calculate new attention Softmax(Q * K)
attention = 0

# See if there is a spike of probability on needle tokens 
