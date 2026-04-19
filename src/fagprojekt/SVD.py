from fagprojekt.model import (
    get_kvq,
    get_messages,
    get_true_attention_values
)

from fagprojekt.evaluate import (
    compare_attention
)

import torch
import numpy as np
from torch.linalg import svd
import matplotlib.pyplot as plt

def do_SVD(matrix, name= None, plot=False):
    """Compute singular value decomposition and optionally plot explained variance.

    Args:
        matrix: Input tensor to decompose.
        name: Label used in plot title and output filename.
        plot: Whether to save a cumulative explained variance plot.

    Returns:
        U, singular values, and Vh from the SVD.
    """
    # CUDA SVD does not support half precision; upcast for decomposition.
    svd_input = matrix.to(torch.float32)
    U, s, Vt = svd(svd_input)

    if plot:
        plt.figure()
        plt.plot(np.cumsum((s**2/sum(s**2)).detach().cpu().numpy()))
        plt.title(f'Explained variance by principal components of {name}')
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.savefig(f'reports/figures/explained_var_{name}.pdf')
    return U, s, Vt


def method_1(key_head, query_head, value_head, k=50):
    """Method 1: Decomposition of the key matrix only"""
    U_K, s_K, Vt_K = do_SVD(key_head)

    # Truncate and calculate K again
    K = U_K[:, :k] @ torch.diag(s_K[:k]) @ Vt_K[:k, :]

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)

    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ K.T)), dim=-1) @ value_head
    return attn_values

def method_2(key_head, query_head, value_head, k=50):
    """Method 2: Decomposition of the key and value matrix seperately"""
    U_K, s_K, Vt_K = do_SVD(key_head)
    U_V, s_V, Vt_V = do_SVD(value_head)

    # Truncate and calculate K & V again
    K = U_K[:, :k] @ torch.diag(s_K[:k]) @ Vt_K[:k, :]
    V = U_V[:, :k] @ torch.diag(s_V[:k]) @ Vt_V[:k, :]

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)

    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ K.T)), dim=-1) @ V
    return attn_values

def method_3(key_head, query_head, value_head, k=50):
    """Method 3: Jointly decompose key and value matrix"""
    # Stack features horizontally:
    joint = torch.cat((key_head, value_head), dim=1)
    U_J, s_J, Vt_J = do_SVD(joint)

    # Truncate matrices
    U_k = U_J[:, :k]
    S_k = torch.diag(s_J[:k])
    Vt_k = Vt_J[:k, :]

    # Determine A matrix
    A = U_k @ S_k  

    # Extract B, C matrix
    head_dim = key_head.shape[1]
    B = Vt_k[:, :head_dim].T
    C = Vt_k[:, head_dim:].T

    # Calculate K and V again
    K = A @ B.T
    V = A @ C.T

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)

    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ K.T)), dim=-1) @ V
    return attn_values


path = "document-haystack/AIG/AIG_5Pages/Text_TextNeedles/AIG_5Pages_TextNeedles_page_4.txt"
messages, prompt, needle = get_messages(path, num_tokens=100)

key_head, value_head, query_head = get_kvq(messages, layer_idx=0, head_idx=0, want_print=True)

attn_values_method_1 = method_1(key_head, query_head, value_head, k=50)
attn_values_method_2 = method_2(key_head, query_head, value_head, k=50)
attn_values_method_3 = method_3(key_head, query_head, value_head, k=50)

print("Attention values dimension for the 3 SVD methods:", attn_values_method_1.size(), attn_values_method_2.size(), attn_values_method_3.size())

true_attn_values = get_true_attention_values(query_head, key_head, value_head)
print(f"True attention values dimension: {true_attn_values.shape}\n")

compare_attention(true_attn_values, attn_values_method_1, "Method 1")
compare_attention(true_attn_values, attn_values_method_2, "Method 2")
compare_attention(true_attn_values, attn_values_method_3, "Method 3")