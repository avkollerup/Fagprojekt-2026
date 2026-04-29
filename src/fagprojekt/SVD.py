from fagprojekt.model import get_kvq, get_messages, get_true_attention_values
import torch

def do_SVD(matrix):
    """Compute singular value decomposition

    Args:
        matrix: Input tensor to decompose.
        name: Label used in plot title and output filename.

    Returns:
        U, singular values, and Vh from the SVD.
    """
    U, s, Vt = torch.linalg.svd(matrix, full_matrices=False)
    return U, s, Vt


def method_1(key_head, query_head, value_head, k):
    """Method 1: Decomposition of the key matrix only"""
    U_K, s_K, Vt_K = do_SVD(key_head)

    k_eff = min(k, s_K.shape[0])
    K = U_K[:, :k_eff] @ torch.diag(s_K[:k_eff]) @ Vt_K[:k_eff, :]

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)

    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ K.T)), dim=-1) @ value_head
    return attn_values

def decompose_K(key_head, k):
    U_K, s_K, Vt_K = do_SVD(key_head)

    k_eff = min(k, s_K.shape[0])
    U_k = U_K[:, :k_eff]
    S_k = torch.diag(s_K[:k_eff])

    A = U_k @ S_k  
    B = Vt_K[:k_eff, :].T
    
    return A, B


def method_2(key_head, query_head, value_head, k):
    """Method 2: Decomposition of the key and value matrix seperately"""
    U_K, s_K, Vt_K = do_SVD(key_head)
    U_V, s_V, Vt_V = do_SVD(value_head)

    k_k = min(k, s_K.shape[0])
    k_v = min(k, s_V.shape[0])
    K = U_K[:, :k_k] @ torch.diag(s_K[:k_k]) @ Vt_K[:k_k, :]
    V = U_V[:, :k_v] @ torch.diag(s_V[:k_v]) @ Vt_V[:k_v, :]

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)

    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ K.T)), dim=-1) @ V
    return attn_values

def method_3(key_head, query_head, value_head, k):
    """Method 3: Jointly decompose key and value matrix"""
    # Stack features horizontally:
    joint = torch.cat((key_head, value_head), dim=1)
    U_J, s_J, Vt_J = do_SVD(joint)

    k_eff = min(k, s_J.shape[0])
    U_k = U_J[:, :k_eff]
    S_k = torch.diag(s_J[:k_eff])
    Vt_k = Vt_J[:k_eff, :]

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

def compare_attention(true_attn, approx_attn, name):
    """ We have used three metrics: 
        - MSE for raw error (might look small because of small values)
        - Frobenius norm for scale-independent accuracy
        - Cosine similarity to capture structural (attention pattern) similarity"""
    
    mse = torch.mean((true_attn - approx_attn) ** 2).item()
    rel_frob = (torch.norm(true_attn - approx_attn, p="fro") / torch.norm(true_attn, p="fro")).item()
    cos = torch.nn.functional.cosine_similarity(true_attn.flatten(), approx_attn.flatten(), dim=0).item()

    print(f"{name}:")
    print(f"  MSE: {mse:.6e}")
    print(f"  Relative Frobenius error: {rel_frob:.6e}")
    print(f"  Cosine similarity: {cos:.6f}\n")


if __name__ == "__main__":
    # --------------- PARAMETERS --------------
    path = "document-haystack/AIG/AIG_5Pages/Text_TextNeedles/AIG_5Pages_TextNeedles_page_4.txt"
    k = 100
    num_tokens = 100
    layer_idx = 0
    head_idx = 0

    # ---------------- Do the 3 SVD methods ----------------
    messages, prompt, needle = get_messages(path, num_tokens=num_tokens)

    key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False)

    attn_values_method_1 = method_1(key_head, query_head, value_head, k=k)
    attn_values_method_2 = method_2(key_head, query_head, value_head, k=k)
    attn_values_method_3 = method_3(key_head, query_head, value_head, k=k)

    print("Attention values dimension for the 3 SVD methods:", attn_values_method_1.size(),attn_values_method_2.size(),attn_values_method_3.size())

    # --------------- Compare the 3 SVD methods with the true attention values ---------------
    true_attn_values = get_true_attention_values(query_head, key_head, value_head)
    print(f"True attention values dimension: {true_attn_values.shape}\n")

    compare_attention(true_attn_values, attn_values_method_1, "Method 1")
    compare_attention(true_attn_values, attn_values_method_2, "Method 2")
    compare_attention(true_attn_values, attn_values_method_3, "Method 3")