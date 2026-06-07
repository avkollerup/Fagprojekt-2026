from fagprojekt.model import get_kvq, get_messages, get_true_attention_values
import torch
import os
import numpy as np

def do_SVD(matrix):
    """Compute singular value decomposition

    Args:
        matrix: Input tensor to decompose.

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
    return K, attn_values

def decompose_K(key_head, k):
    U_K, s_K, Vt_K = do_SVD(key_head)

    k_eff = min(k, s_K.shape[0])
    U_k = U_K[:, :k_eff]
    S_k = torch.diag(s_K[:k_eff])

    A = U_k @ S_k  
    B = Vt_K[:k_eff, :].T
    
    return A, B

def method_4(key_head, query_head, value_head, k):
    """Method 4: Decomposition of the value matrix only"""
    U_V, s_V, Vt_V = do_SVD(value_head)

    k_eff = min(k, s_V.shape[0])
    V = U_V[:, :k_eff] @ torch.diag(s_V[:k_eff]) @ Vt_V[:k_eff, :]

    # Mask: Strict upper-triangular = -inf above diagonal, 0 on/below diagonal
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)

    # Compute attention values
    attn_values = torch.softmax((M + (query_head @ key_head.T)), dim=-1) @ V
    return V, attn_values

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
    return K, V, attn_values

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
    return K, V, attn_values

def first_k_for_threshold(matrix: torch.Tensor, threshold: float = 0.9) -> int:
    """Smallest k such that the top-k singular values capture at least `threshold` of total variance."""
    singular_values = torch.linalg.svdvals(matrix.float())
    explained_variance = singular_values ** 2
    cumulative_ratio = torch.cumsum(explained_variance, dim=0) / torch.sum(explained_variance)
    indices = torch.where(cumulative_ratio >= threshold)[0]
    return int(indices[0].item()) + 1  # component needed


def tune_threshold(key_head, query_head, value_head, threshold_list, max_rel_mse):
    """Find the most compressed k for each SVD method that keeps relative MSE <= max_rel_mse.

    Sweeps all thresholds in a single pass, collecting every (threshold, k) pair that meets
    the budget and simultaneously tracking the lowest-MSE result as a fallback.

    Fallback when no threshold meets the budget: result with the lowest MSE.
    """
    true_attn = get_true_attention_values(query_head, key_head, value_head)
    # Clamp denominator to avoid division by zero for degenerate all-zero attention outputs
    denom = torch.clamp(torch.mean(true_attn ** 2), min=1e-12).item()
    joint = torch.cat((key_head, value_head), dim=1)

    valid    = {'method_1': [], 'method_2': [], 'method_3': [], 'method_4': []}
    fallback = {'method_1': None, 'method_2': None, 'method_3': None, 'method_4': None}

    for threshold in sorted(threshold_list):
        k_k      = first_k_for_threshold(key_head,  threshold)
        k_v      = first_k_for_threshold(value_head, threshold)
        k_joint  = first_k_for_threshold(joint,      threshold)
        # Method 2 decomposes K and V separately, so both must independently meet the threshold
        k_kv_sep = max(k_k, k_v)

        candidates = {
            'method_1': (k_k,      method_1(key_head, query_head, value_head, k=k_k)[1]),
            'method_4': (k_v,      method_4(key_head, query_head, value_head, k=k_v)[1]),
            'method_2': (k_kv_sep, method_2(key_head, query_head, value_head, k=k_kv_sep)[2]),
            'method_3': (k_joint,  method_3(key_head, query_head, value_head, k=k_joint)[2]),
        }

        for name, (k, approx) in candidates.items():
            mse = torch.mean((true_attn - approx) ** 2).item()
            rel_mse = mse / denom
            if rel_mse <= max_rel_mse:
                valid[name].append({'threshold': threshold, 'k': k, 'rel_mse': rel_mse})
            # Always track the lowest-MSE result in case no threshold meets the budget
            if fallback[name] is None or mse < fallback[name]['mse']:
                fallback[name] = {'threshold': threshold, 'k': k, 'rel_mse': rel_mse, 'mse': mse}

    best = {}
    for name in valid:
        if valid[name]:
            result = min(valid[name], key=lambda x: (x['k'], x['rel_mse']))
            best[name] = {'threshold': result['threshold'], 'k': result['k'], 'rel_mse': result['rel_mse'], 'met_budget': True}
        else:
            fb = fallback[name]
            best[name] = {'threshold': fb['threshold'], 'k': fb['k'], 'rel_mse': fb['rel_mse'], 'met_budget': False}

    return best


def compare_attention(true_attn, approx_attn, name, want_print=True):
    """ We have used three metrics:
        - MSE for raw error (might look small because of small values)
        - Frobenius norm for scale-independent accuracy
        - Cosine similarity to capture structural (attention pattern) similarity"""

    mse = torch.mean((true_attn - approx_attn) ** 2).item()
    rel_frob = (torch.norm(true_attn - approx_attn, p="fro") / torch.norm(true_attn, p="fro")).item()
    cos = torch.nn.functional.cosine_similarity(true_attn.flatten(), approx_attn.flatten(), dim=0).item()
    if want_print:
        print(f"{name}:")
        print(f"  MSE: {mse:.6e}")
        print(f"  Relative Frobenius error: {rel_frob:.6e}")
        print(f"  Cosine similarity: {cos:.6f}\n")
    return mse, rel_frob, cos


if __name__ == "__main__":
    """Experiment: threshold tuning for SVD-based attention compression
    
    Goal: find the smallest threshold of explained variance per method
    that keeps the attention output error within an acceptable budget.
    
    Why threshold instead of k directly?
       k is hard to set without context. The SVD threshold is more interpretable:
       it controls what fraction of singular value variance is retained, so
       threshold=0.9 means "keep enough components to explain 90% of the variance."
       tune_threshold converts this into a concrete k by finding the first index
       where the cumulative explained variance crosses the threshold.
    
    Why sweep multiple rel_mse budgets?
       The budget max_rel_mse = MSE / mean(true²) asks "how much attention error am I willing to tolerate?" 
       Sweeping a range reveals at what budget each method becomes viable and how aggressively k
       can be reduced while staying within the budget. If no threshold meets the budget, 
       tune_threshold falls back to the threshold with the lowest MSE so results are always interpretable.
    
    Final comparison: each method is run at the threshold achieving its lowest
    rel_mse across the full sweep (budget-free best quality), giving an upper
    bound on how well each method can approximate the true attention output."""

    from dotenv import load_dotenv
    load_dotenv()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    path = "document-haystack/AIG/AIG_5Pages/Text_TextNeedles/AIG_5Pages_TextNeedles_page_4.txt"
    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx  = int(os.environ["LAYER_IDX"])
    head_idx   = int(os.environ["HEAD_IDX"])

    messages, prompt, needle = get_messages(path, num_tokens=num_tokens)
    key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False)
    true_attn_values = get_true_attention_values(query_head, key_head, value_head)
    print(f"True attention values dimension: {true_attn_values.shape}\n")

    threshold_list = np.linspace(0.7, 0.99, 50).tolist()
    rel_mse_budgets = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    for max_rel_mse in rel_mse_budgets:
        tuned = tune_threshold(key_head, query_head, value_head, threshold_list, max_rel_mse)
        print(f"--- max_rel_mse={max_rel_mse} ---")
        for name, result in tuned.items():
            status = "ok" if result['met_budget'] else "budget not met"
            print(f"  {name}: threshold={result['threshold']:.3f}, k={result['k']}, rel_mse={result['rel_mse']:.6e} ({status})")
        print()

    # max_rel_mse=0 forces all methods to fallback, returning the lowest-MSE result per method
    best = tune_threshold(key_head, query_head, value_head, threshold_list, max_rel_mse=0)
    _, attn_values_method_1   = method_1(key_head, query_head, value_head, k=best['method_1']['k'])
    _, _, attn_values_method_2 = method_2(key_head, query_head, value_head, k=best['method_2']['k'])
    _, _, attn_values_method_3 = method_3(key_head, query_head, value_head, k=best['method_3']['k'])
    _, attn_values_method_4   = method_4(key_head, query_head, value_head, k=best['method_4']['k'])

    print("--- Best approximation per method (lowest rel_mse across sweep) ---")
    for name, result in best.items():
        print(f"  {name}: threshold={result['threshold']:.3f}, k={result['k']}, rel_mse={result['rel_mse']:.6e}")
    print()

    print("Attention approximation error at each method's best threshold:")
    compare_attention(true_attn_values, attn_values_method_1, "Method 1: Decompose only K")
    compare_attention(true_attn_values, attn_values_method_4, "Method 4: Decompose only V")
    compare_attention(true_attn_values, attn_values_method_2, "Method 2: Decompose K and V separately")
    compare_attention(true_attn_values, attn_values_method_3, "Method 3: Decompose K and V jointly")