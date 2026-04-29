# # UDKOMMENTER for kun at bruge gpu 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from fagprojekt.SVD import decompose_K, method_1, compare_attention
from fagprojekt.model import get_kvq, get_messages


def build_mlp(seq_len,k):
    """A small feed-forward network."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(seq_len*k, 784),
        nn.ReLU(),
        nn.Linear(784, seq_len*k),
        nn.Unflatten(0,(seq_len,k)),
    )


def hokus_pokus(query_head, value_head, a_mat, b_mat, method, g_theta):
    """Approximate attention values with optional learned score transform.

    Args:
        query_head: Query tensor of shape [seq_len, head_dim].
        value_head: Value tensor of shape [seq_len, head_dim].
        a_mat: Decomposition matrix A with shape [seq_len, rank].
        b_mat: Decomposition matrix B with shape [head_dim, rank].
        method: Either "identity" or "mlp".
        g_theta: Optional learnable module used when method == "mlp".

    Returns:
        Approximated attention values with shape [seq_len, head_dim].
    """

    # QB
    print(query_head.size())
    print(b_mat.size())
    input_data = (query_head @ b_mat)
    
    print(input_data.size())
    # g(QB), where g is the identity function
    if method == "identity":
        transformed_input_data = input_data

    # g(QB), where g is an MLP
    elif method == "mlp":
        transformed_input_data = g_theta(input_data)

    # Hokus pokus: g(QB)A^TV
    return transformed_input_data @ a_mat.T @ value_head



def train(path, method="mlp", epochs = 500, lr = 1e-3, k = 50):
    """Train g_theta to mimic the baseline approximation.

    Args:
        path: Path to the page from Needle-in-a-Haystack to use in the prompt
        method:  "mlp".
        epochs: Number of optimization steps.
        lr: Learning rate for Adam.
        k: Rank used for K decomposition.

    Returns:
        The trained g_theta module.
    """

    if method != "mlp":
        raise ValueError("Method not mlp")

    messages, _, _ = get_messages(path, num_tokens=100)

    key_head, value_head, query_head = get_kvq(messages, layer_idx=0, head_idx=0, want_print=True)
    true_attn = method_1(key_head, query_head, value_head, k=k).detach()

    # Perform SVD decomposition of K to get A and B matrices
    a_mat, b_mat = decompose_K(key_head, k=k)

    seq_len = query_head.shape[0]
    g_theta = build_mlp(seq_len,k).to(query_head.device)
    optimizer = torch.optim.Adam(g_theta.parameters(), lr=lr)

    train_loss = []

    for step in range(epochs):
        optimizer.zero_grad()

        y_pred = hokus_pokus(
            query_head=query_head,
            value_head=value_head,
            a_mat=a_mat,
            b_mat=b_mat,
            method=method,
            g_theta=g_theta,
        )

        loss = torch.nn.functional.mse_loss(true_attn, y_pred)
        
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.6e}")

    # Plotting
    output_path = f"reports/figures/hokus_pokus_train_loss_{method}.png"

    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, linewidth=2)
    plt.title("Hokus Pokus Training Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved train-loss plot to {output_path}")

    return g_theta


def compare_hokus_pokus(path, method, model_path, k=50):
    """Load a saved model checkpoint and compare it against true attention.

    Args:
        path: Path to the page from Needle-in-a-Haystack used in the prompt.
        method: Either "identity" or "mlp".
        model_path: Path to the saved model checkpoint.
        k: Rank used for K decomposition.

    Returns:
        The comparison output tensor.
    """
    messages, _, _ = get_messages(path, num_tokens=100)
    key_head, value_head, query_head = get_kvq(messages, layer_idx=0, head_idx=0, want_print=True)
    true_attn = method_1(key_head, query_head, value_head, k=k).detach()
    
    # Perform SVD decomposition of K to get A and B matrices
    a_mat, b_mat = decompose_K(key_head, k=k)

    # scale a_mat and b_mat to ensure that the scale does not explode
    alpha = torch.linalg.norm(a_mat, axis=1)
    a_mat = a_mat @ torch.linalg.inv(torch.diag(alpha))
    b_mat = torch.diag(alpha) @ b_mat

    if method == "identity":
        loaded_g_theta = None
    
    elif method == "mlp":
        seq_len = query_head.shape[0]
        loaded_g_theta = build_mlp(seq_len).to(query_head.device)
        loaded_g_theta.load_state_dict(torch.load(model_path, map_location=query_head.device))
        loaded_g_theta.eval()

    else:
        raise ValueError("Method not mlp or identity")

    final_attn = hokus_pokus(
        query_head=query_head,
        value_head=value_head,
        a_mat=a_mat,
        b_mat=b_mat,
        method=method,
        g_theta=loaded_g_theta,
    ).detach()
    compare_attention(true_attn, final_attn, "Hokus Pokus vs true_attn")

    return final_attn


if __name__ == "__main__":
    path = "document-haystack/AIG/AIG_5Pages/Text_TextNeedles/AIG_5Pages_TextNeedles_page_4.txt"
    method = "mlp"

    if method == "identity":
        compare_hokus_pokus(path=path, method=method, model_path=None, k=50)
    else:
        # Train model
        g_theta = train(path=path, method=method)

        # Save model
        model_path = f"models/g_theta_weights_{method}.pth"
        torch.save(g_theta.state_dict(), model_path)

        # Load and compare model   
        compare_hokus_pokus(path=path, method=method, model_path=model_path, k=50)
