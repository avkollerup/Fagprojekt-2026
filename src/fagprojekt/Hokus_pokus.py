import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from fagprojekt.SVD import decompose_K, method_1
from fagprojekt.model import get_kvq, get_messages


def build_mlp(feature_dim):
    """Build a simple MLP for row-wise score transformation.

    Args:
        feature_dim: Number of features per row.

    Returns:
        A small feed-forward network.
    """
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(),
        nn.Linear(feature_dim, feature_dim),
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
    seq_len = query_head.shape[0]
    M = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)

    input_data = M + (query_head @ b_mat)

    if method == "identity":
        transformed_input_data = input_data
    elif method == "mlp":
        transformed_input_data = g_theta(input_data)

    # weights = torch.softmax(transformed_input_data, dim=-1)
    return transformed_input_data @ a_mat.T @ value_head





def train(path, method, epochs = 500, lr = 1e-3, k = 50):
    """Train g_theta to mimic the baseline approximation.

    Args:
        path: Path to the page from Needle-in-a-Haystack to use in the prompt
        method: Either "identity" or "mlp".
        epochs: Number of optimization steps.
        lr: Learning rate for Adam.
        k: Rank used for K decomposition.

    Returns:
        The trained g_theta module.
    """
    messages, _, _ = get_messages(path, num_tokens=100)

    key_head, value_head, query_head = get_kvq(messages, layer_idx=0, head_idx=0, want_print=True)
    true_attn = method_1(key_head, query_head, value_head, k=k).detach()

    a_mat, b_mat = decompose_K(key_head, k=k)

    seq_len = query_head.shape[0]
    g_theta = build_mlp(seq_len).to(query_head.device)
    optimizer = torch.optim.Adam(g_theta.parameters(), lr=lr)
    loss_fn = torch.nn.functional.mse_loss()

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

        loss = loss_fn(true_attn, y_pred)
        
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.6e}")

    # Plotting
    output_path = "reports/figures/hokus_pokus_train_loss.png"

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


if __name__ == "__main__":
    path = "document-haystack/AIG/AIG_5Pages/Text_TextNeedles/AIG_5Pages_TextNeedles_page_4.txt"
    g_theta = train(path=path, method="mlp")
    # Save model
    torch.save(g_theta.state_dict(), "models/g_theta_weights.pth")
