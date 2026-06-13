# # UDKOMMENTER for kun at bruge gpu 0
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.profiler import ProfilerAction, profile
import math
from fagprojekt.SVD import decompose_K, compare_attention
from fagprojekt.model import get_kvq, get_messages, load_model, get_true_attention_values
import random

prof = profile()

def build_mlp(k):
    """A small feed-forward network."""
    return nn.Sequential(
        nn.Linear(k, 784),
        nn.ReLU(),
        nn.Linear(784, k),
        nn.Softmax(dim=-1),
        nn.Linear(k, k)
    )


def hokus_pokus(query_head, value_head, key_head, k, method, g_theta):
    """Approximate attention values with optional learned score transform.

    Args:
        query_head: Query tensor of shape [seq_len, head_dim].
        value_head: Value tensor of shape [seq_len, head_dim].
        key_head: Key tensor of shape [seq_len, head_dim].
        k: Rank used for K decomposition.
        a_mat: Decomposition matrix A with shape [seq_len, rank].
        b_mat: Decomposition matrix B with shape [head_dim, rank].
        method: Either "identity" or "mlp".
        g_theta: Optional learnable module used when method == "mlp".

    Returns:
        Approximated attention values with shape [seq_len, head_dim].
    """

    # Perform SVD decomposition of K to get A and B matrices
    a_mat, b_mat = decompose_K(key_head, k=k)
    # Detach to break computation graph (a_mat, b_mat are fixed, not trainable)
    a_mat = a_mat.detach()
    b_mat = b_mat.detach()

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


def train(paths, method="mlp", lr = 1e-3, k = 50, layer_idx=0, head_idx=0,loss_method='cosine',plot_figure=True,model=None, tokenizer=None,tokens=100, num_epochs=10):
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

    # initialize g_theta, optimizer and loss function on CPU to reduce GPU memory pressure
    compute_device = torch.device("cpu")
    g_theta = build_mlp(k).to(compute_device)
    optimizer = torch.optim.Adam(g_theta.parameters(), lr=lr)
    if loss_method == 'cosine':
        loss_fn = torch.nn.CosineEmbeddingLoss()
    elif loss_method == 'mse':
        loss_fn = torch.nn.MSELoss()
    
    train_loss = []
    best_loss = float('inf')
    best_state = None
    patience = 10
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        random.shuffle(paths)
        epoch_loss = []
        for path in paths:

            # get messages for path
            messages, _, _ = get_messages(path, num_tokens=tokens)

            # Use no_grad to prevent graph tracking from model inference
            with torch.no_grad():
                key_head, value_head, query_head = get_kvq(
                    messages,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    want_print=False,
                    model=model,
                    tokenizer=tokenizer,
                )

            # Move extracted head tensors to CPU as early as possible to reduce GPU memory pressure.
            key_head = key_head.cpu()
            value_head = value_head.cpu()
            query_head = query_head.cpu()
            torch.cuda.empty_cache()

            with torch.no_grad():
                true_attn = get_true_attention_values(
                    query_head=query_head,
                    key_head=key_head,
                    value_head=value_head,
                ).detach()

            y_pred = hokus_pokus(
                query_head=query_head,
                value_head=value_head,
                key_head=key_head,
                k=k,
                method=method,
                g_theta=g_theta,
            )

            # compute loss based on method
            if loss_method == 'cosine':
                loss = loss_fn(y_pred.flatten(), true_attn.flatten(), torch.tensor(1, device=y_pred.device))
            elif loss_method == 'mse':
                loss = loss_fn(y_pred, true_attn)

            epoch_loss.append(loss.clone().item())
            train_loss.append(loss.clone().item())

            # zero the gradients before backward pass
            optimizer.zero_grad()

            # step and backwardpass
            loss.backward()
            optimizer.step()

            del key_head, value_head, query_head, true_attn, y_pred
            torch.cuda.empty_cache()

        mean_epoch_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, mean loss={mean_epoch_loss:.6e}")

        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss
            best_state = {k: v.clone() for k, v in g_theta.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                g_theta.load_state_dict(best_state)
                break

    # Plotting
    if plot_figure:
        output_path = f"reports/figures/hokus_pokus_train_loss_{method}_k_{k}_epochs_{num_epochs}.png"

        plt.figure(figsize=(8, 4))
        plt.plot(train_loss, linewidth=2)
        plt.title("Hokus Pokus Training Loss")
        plt.xlabel("Step")
        plt.ylabel("1 - Cosine Similarity" if loss_method == "cosine" else "MSE Loss")
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
    rmse_errors=[]
    frob_norm_errors=[]
    cosine_errors=[]

    for path in paths:
        messages, _, _ = get_messages(path, num_tokens=tokens)

        # Use no_grad to prevent graph tracking from model inference
        with torch.no_grad():
            key_head, value_head, query_head = get_kvq(
                messages,
                layer_idx=layer_idx,
                head_idx=head_idx,
                want_print=False,
                model=model,
                tokenizer=tokenizer,
            )

        if loaded_g_theta is not None:
            compute_device = next(loaded_g_theta.parameters()).device
        else:
            compute_device = torch.device("cpu")

        key_head = key_head.to(compute_device)
        value_head = value_head.to(compute_device)
        query_head = query_head.to(compute_device)
        torch.cuda.empty_cache()

        with torch.no_grad():
            true_attn = get_true_attention_values(
                query_head=query_head,
                key_head=key_head,
                value_head=value_head,
            ).detach()

            if method == "identity":
                loaded_g_theta = None
            
            elif method == "mlp":
                # if we have already loaded g_theta, no need to load it again
                if loaded_g_theta is None:
                    loaded_g_theta = build_mlp(k).to(compute_device)
                    loaded_g_theta.load_state_dict(torch.load(model_path, map_location=compute_device))
                    loaded_g_theta.eval()

            else:
                raise ValueError("Method not mlp or identity")

            final_attn = hokus_pokus(
                query_head=query_head,
                value_head=value_head,
                key_head=key_head,
                k=k,
                method=method,
                g_theta=loaded_g_theta,
            ).clone()

        mse,frob,cos = compare_attention(true_attn, final_attn, "Hokus Pokus vs true_attn",want_print=False)
        # append errors to lists for later analysis
        rmse_errors.append(math.sqrt(mse))
        frob_norm_errors.append(frob)
        cosine_errors.append(cos)

        del key_head, value_head, query_head, true_attn, final_attn
        torch.cuda.empty_cache()
    # after loop, compute average errors
    avg_rmse = sum(rmse_errors) / len(rmse_errors)
    avg_frob = sum(frob_norm_errors) / len(frob_norm_errors)
    avg_cos = sum(cosine_errors) / len(cosine_errors)

    return (avg_rmse, avg_frob, avg_cos, rmse_errors)

 
def k_fold_crossvalidation_decide_k(model = None, tokenizer = None,folds=9,layer_idx=0,head_idx=0,num_tokens=200,method = 'mse'):
    """Perform 9-fold crossvalidation to determine
    the best number of components to keep in the decomposition of K
    in the Hokus Pokus method"""
    k_values = range(10,105,5)
    # Load model if it does not exist yet
    if (model == None) or (tokenizer == None):
        model,tokenizer = load_model(want_print=False)
    if folds != 9:
        raise ValueError("This method is only implemented for 9-fold crossvalidation." \
        f"You are currently using {folds}. Please change to 9. Hilsen Elisabeth")
    
    companies = ['Barclays','BlackRock','BNYMellon','CapitalOne','CitiGroup','Cofinimmo','CVS','DWS','Entain']
    for fold in range(folds):
        print(f'Starting fold {fold+1}')
        # get paths
        train_paths = []
        for i in range(5):
            if i == fold:
                test_dir = Path(f"document-haystack/{companies[fold]}/{companies[fold]}_25Pages/Text_TextNeedles")
                test_paths = list(test_dir.glob("*.txt"))
                test_paths = [str(p) for p in test_paths]
            else:
                base_dir = Path(f"document-haystack/{companies[i]}/{companies[i]}_25Pages/Text_TextNeedles")
                paths = list(base_dir.glob("*.txt"))
                train_paths.extend([str(p) for p in paths])
        
        # now train for evey k value:
        mses = {k_value:[] for k_value in k_values}
        for k_val in k_values:
            print(f'Training and evaluating model with k={k_val}')
            # Train model on the training paths
            g_theta = train(train_paths, method='mlp',layer_idx=layer_idx,head_idx=head_idx,k=k_val,model=model,tokenizer=tokenizer,loss_method=method,tokens=num_tokens,plot_figure=False)
            g_theta.eval()

            # Load and compare model   
            with torch.no_grad():
                mse,frob,cos,_ = compare_hokus_pokus(paths=test_paths, method='mlp', loaded_g_theta=g_theta, k=k_val,layer_idx=layer_idx, head_idx=head_idx,tokens=num_tokens,model=model, tokenizer=tokenizer)
            mses[k_val].append(np.sqrt(mse))
            
            # cleanup
            del g_theta
            torch.cuda.empty_cache()

    
    # now we should compute the average rmse for each k over folds
    for k_val in k_values:
        mses[k_val] = sum(mses[k_val])/folds
    plt.plot(k_values,list(mses.values()),'o-')
    plt.title('RMSE error on test set over k values')
    plt.xlabel('K')
    plt.ylabel('RMSE error on attention')
    plt.savefig('reports/figures/hokus_pokus_k_analysis_CV.png')

    return mses

def get_rmse_companies_Hokus_Pokus(model, tokenizer, num_tokens, layer_idx, head_idx, k, train_companies, test_companies, num_epochs):
    # --------------- TRAINING AND TEST PATHS ---------------
    train_paths = [f'{Path(f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles")}_page_{page}.txt' for company in train_companies for page in range(1,26)]
    test_paths = [f'{Path(f"document-haystack/{company}/{company}_25Pages/Text_TextNeedles/{company}_25Pages_TextNeedles")}_page_{page}.txt' for company in test_companies for page in range(1,26)]

    method = "mlp"
    loss_method = 'mse'

    print(f"Training with loss method: {loss_method}")
    
    # Train model on the training paths
    g_theta = train(train_paths, method=method, layer_idx=layer_idx, head_idx=head_idx, k=k, model=model, tokenizer=tokenizer, loss_method=loss_method, tokens=num_tokens, plot_figure=True, num_epochs=num_epochs)

    # Save model
    model_path = f"models/g_theta_weights_{method}_k_{k}_epochs_{num_epochs}.pth"
    torch.save(g_theta.state_dict(), model_path)

    # load the g_theta model weights only once
    g_theta_loaded = build_mlp(k).to(next(g_theta.parameters()).device)
    g_theta_loaded.load_state_dict(torch.load(model_path, map_location=next(g_theta.parameters()).device))
    g_theta_loaded.eval()

    # Load and compare model
    _, _, _, rmse_per_page = compare_hokus_pokus(paths=test_paths, method=method, model_path=model_path, loaded_g_theta=g_theta_loaded, k=k, layer_idx=layer_idx, head_idx=head_idx, tokens=num_tokens, model=model, tokenizer=tokenizer)
    return rmse_per_page



if __name__ == "__main__":
    # --------------- PARAMETERS --------------
    from dotenv import load_dotenv
    load_dotenv()

    num_tokens = int(os.environ["NUM_TOKENS"])
    layer_idx = int(os.environ["LAYER_IDX"])
    head_idx = int(os.environ["HEAD_IDX"])
    #k =  int(os.environ["k"]) # number of components to keep in SVD decomposition

    # --------------- LOAD MODEL ONLY ONCE ---------------
    print("Loading model and tokenizer once at the start...")
    model, tokenizer = load_model(want_print=False)

    mses = k_fold_crossvalidation_decide_k(model=model,tokenizer=tokenizer,layer_idx=layer_idx,head_idx=head_idx,num_tokens=num_tokens,method='mse')
    
    k_values = range(10,105,5)
    print(k_values)
    print(mses)