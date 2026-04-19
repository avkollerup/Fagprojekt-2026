import torch

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