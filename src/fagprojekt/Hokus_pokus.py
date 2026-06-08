# # UDKOMMENTER for kun at bruge gpu 0
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from fagprojekt.SVD import decompose_K, compare_attention
from fagprojekt.model import get_kvq, get_messages, load_model, get_true_attention_values


def build_mlp(k):
    """A small feed-forward network."""
    return nn.Sequential(
        nn.Linear(k, 784),
        nn.ReLU(),
        nn.Linear(784, k),
        nn.Softmax(dim=-1),
        nn.Linear(k, k)
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
    # print(query_head.size())
    # print(b_mat.size())
    input_data = query_head @ b_mat
    
    # print(input_data.size())
    # g(QB), where g is the identity function
    if method == "identity":
        transformed_input_data = input_data

    # g(QB), where g is an MLP
    elif method == "mlp":
        transformed_input_data = g_theta(input_data)

    # Hokus pokus: g(QB)A^TV
    return transformed_input_data @ a_mat.T @ value_head



def train(paths, method="mlp", lr = 1e-3, k = 50, layer_idx=0, head_idx=0,loss_method='cosine',plot_figure=True,model=None, tokenizer=None):
    """Train g_theta to mimic the baseline approximation.

    Args:
        path: Path to the page from Needle-in-a-Haystack to use in the prompt
        method:  "mlp".
        lr: Learning rate for Adam.
        k: Rank used for K decomposition.

    Returns:
        The trained g_theta module.
    """

    if method != "mlp":
        raise ValueError("Method not mlp")
    
    # load model and tokenizer once to avoid doing it every step
    if model is None or tokenizer is None:
        model, tokenizer = load_model(want_print=False)
    # initialize g_theta, optimizer and loss function
    g_theta = build_mlp(k).to(model.device)
    optimizer = torch.optim.Adam(g_theta.parameters(), lr=lr)
    if loss_method == 'cosine':
        loss_fn = torch.nn.CosineEmbeddingLoss()
    elif loss_method == 'mse':
        loss_fn = torch.nn.MSELoss()
    
    train_loss = []

    step=0
    for path in paths:

        # get messages for path
        messages, _, _ = get_messages(path, num_tokens=100)

        # Use no_grad to prevent graph tracking from model
        # extract the heads and true attention values
        with torch.no_grad():
            key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, 
                                                       want_print=False, model=model, tokenizer=tokenizer)

        "_____________________KOMMENTAR_________________"
        "SKAL key_head og query_head ikke byttes om her?"
        true_attn = get_true_attention_values(key_head, query_head, value_head).detach()

        # Perform SVD decomposition of K to get A and B matrices
        a_mat, b_mat = decompose_K(key_head, k=k)
        # Detach to break computation graph (a_mat, b_mat are fixed, not trainable)
        a_mat = a_mat.detach()
        b_mat = b_mat.detach()

        y_pred = hokus_pokus(
            query_head=query_head,
            value_head=value_head,
            a_mat=a_mat,
            b_mat=b_mat,
            method=method,
            g_theta=g_theta,
        )


        # compute loss based on method
        if loss_method == 'cosine':
            loss = loss_fn(y_pred.flatten(), true_attn.flatten(), torch.tensor(1, device=y_pred.device))
        elif loss_method == 'mse':
            loss = loss_fn(y_pred, true_attn)
        # elif loss_method == 'cosine_manual':
        #     cos_sim = torch.nn.functional.cosine_similarity(
        #         true_attn.flatten(),y_pred.flatten(), dim=0)
        #     loss = 1 - cos_sim  # Minimize this to maximize similarity
        train_loss.append(loss.clone().item())
        
        # zero the gradients before backward pass
        optimizer.zero_grad()

        # step and backwardpass
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"step={step} loss={train_loss[-1]:.6e}")
        step += 1

    # Plotting
    if plot_figure:
        output_path = f"reports/figures/hokus_pokus_train_loss_{method}.png"

        plt.figure(figsize=(8, 4))
        plt.plot(train_loss, linewidth=2)
        plt.title("Hokus Pokus Training Loss")
        plt.xlabel("Step")
        plt.ylabel("1 - Cosine Similarity")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved train-loss plot to {output_path}")


    return g_theta


def compare_hokus_pokus(paths, method, model_path=None, loaded_g_theta=None, model=None, tokenizer=None, k=50,layer_idx=0, head_idx=0,tokens=100):
    """Load a saved model checkpoint and compare it against true attention.

    Args:
        path: Path to the page from Needle-in-a-Haystack used in the prompt.
        method: Either "identity" or "mlp".
        model_path: Path to the saved model checkpoint.
        k: Rank used for K decomposition.

    Returns:
        The comparison output tensor.
    """
    mse_errors=[]
    frob_norm_errors=[]
    cosine_errors=[]

    for path in paths:
        messages, _, _ = get_messages(path, num_tokens=100)

        # Use no_grad to prevent graph tracking from model
        with torch.no_grad():
            key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False, model=model, tokenizer=tokenizer)

        "_____________________KOMMENTAR_________________"
        "SKAL key_head og query_head ikke byttes om her?"
        true_attn = get_true_attention_values(key_head, query_head, value_head).detach()
        
        # Perform SVD decomposition of K to get A and B matrices
        a_mat, b_mat = decompose_K(key_head, k=k)
        # Detach to break computation graph
        a_mat = a_mat.detach()
        b_mat = b_mat.detach()


        if method == "identity":
            loaded_g_theta = None
        
        elif method == "mlp":
            # if we have already loaded g_theta, no need to load it again
            if loaded_g_theta is None:
                loaded_g_theta = build_mlp(k).to(query_head.device)
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
        ).clone()
        mse,frob,cos = compare_attention(true_attn, final_attn, "Hokus Pokus vs true_attn",want_print=False)
        # append errors to lists for later analysis
        mse_errors.append(mse)
        frob_norm_errors.append(frob)
        cosine_errors.append(cos)
    # after loop, compute average errors
    avg_mse = sum(mse_errors) / len(mse_errors)
    avg_frob = sum(frob_norm_errors) / len(frob_norm_errors)
    avg_cos = sum(cosine_errors) / len(cosine_errors)

    return (avg_mse, avg_frob, avg_cos)


if __name__ == "__main__":
    # --------------- PARAMETERS --------------
    from dotenv import load_dotenv
    load_dotenv()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])
    k =  int(os.environ["k"]) # number of components to keep in SVD decomposition

    # --------------- LOAD MODEL ONLY ONCE ---------------
    print("Loading model and tokenizer once at the start...")
    model, tokenizer = load_model(want_print=False)

    # --------------- TRAINING AND TEST PATHS ---------------
    # create list of paths for training
    base_dir = Path("document-haystack/AIG/AIG_25Pages/Text_TextNeedles")
    paths = list(base_dir.glob("*.txt"))
    paths = [str(p) for p in paths]

    # test paths which are unseen during training
    test_dir = Path("document-haystack/AmericanAirlines/AmericanAirlines_10Pages/Text_TextNeedles")
    test_paths = list(test_dir.glob("*.txt"))
    test_paths = [str(p) for p in test_paths]


    # define method
    method = "mlp"

    # if we just use the identity method, there is no need for training
    if method == "identity":
        compare_hokus_pokus(paths=test_paths, method=method, model_path=None, k=k)

    else:
        # compute for all types of loss methods
        for loss_method in ['mse', 'cosine']:
            print(f"Training with loss method: {loss_method}")

            "_____________________KOMMENTAR_________________"
            "LOSS METODERNE SKAL VEL GIVES SOM INPUT TIL TRAIN? ELLERS BRUGER DEN BARE COSINE"
            # Train model on the training paths
            g_theta = train(paths, method=method,layer_idx=layer_idx,head_idx=head_idx,k=k,model=model,tokenizer=tokenizer,loss_method=loss_method)

            # Save model
            model_path = f"models/g_theta_weights_{method}.pth"
            torch.save(g_theta.state_dict(), model_path)

            # load the g_theta model weights only once
            g_theta_loaded = build_mlp(k).to(next(g_theta.parameters()).device)
            g_theta_loaded.load_state_dict(torch.load(model_path, map_location=next(g_theta.parameters()).device))
            g_theta_loaded.eval()

            # Load and compare model   
            mse,frob,cos = compare_hokus_pokus(paths=test_paths, method=method, model_path=model_path, loaded_g_theta=g_theta_loaded, k=k,layer_idx=layer_idx, head_idx=head_idx,tokens=num_tokens,model=model, tokenizer=tokenizer)
            print(f"Final comparison for loss_method={loss_method}:")
            print(f"  Average MSE: {mse:.6e}")
            print(f"  Average Relative Frobenius error: {frob:.6e}")
            print(f"  Average Cosine similarity: {cos:.6f}\n")
