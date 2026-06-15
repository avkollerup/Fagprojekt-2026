from fagprojekt.SVD import do_SVD
from fagprojekt.model import (get_kvq, _get_tokenizer)
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
import re
import random
from pathlib import Path

def find_token_positions(tokenizer, messages, needle):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # take messages (system + user + assistant format) and converts it into the exact token sequence the model sees
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,).to(device)

    # extract the full token sequence as a list
    input_ids = inputs["input_ids"][0].tolist()

    # tokenize just the needle string
    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]

    # sliding window looking for exact needle match over token IDs
    for i in range(len(input_ids) - len(needle_ids) + 1):
        if input_ids[i:i + len(needle_ids)] == needle_ids:
            return list(range(i, i + len(needle_ids)))

    return []



# Computes both attention weights and final attention output
def get_attention_output(query_head, key_head, value_head):
    # Define dimensions
    d = key_head.shape[-1]

    # Create mask to prevent attending to future tokens
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)

    # Compute attention weights A = softmax(M + QK^T / sqrt(d))
    A = torch.softmax(M + (query_head @ key_head.T) / math.sqrt(d), dim=-1)

    # Compute attention output O = A @ V
    O = A @ value_head

    return A, O



def evaluate_head(query_head, key_true, value_true, key_approx, value_approx, needle_positions):
    # Compute true attention weights and outputs using original K and V
    A_true, O_true = get_attention_output(query_head, key_true, value_true)

    # Compute attention weights and outputs using approximated K and V
    A_approx, O_approx = get_attention_output(query_head, key_approx, value_approx)

    # Use the last query token
    q_idx = -1

    # Sum attention placed on the needle tokens
    true_needle_attention = A_true[q_idx, needle_positions].sum()
    approx_needle_attention = A_approx[q_idx, needle_positions].sum()

    # Compare using cosine similarity
    cosine = F.cosine_similarity(
        O_true[q_idx].unsqueeze(0), # make 1D tensor to 2D 
        O_approx[q_idx].unsqueeze(0), # make 1D tensor to 2D 
        dim=-1,
    )

    output_path = "reports/tables/top_needle_heads.txt"

    with open(output_path, "a") as f:
        f.write("---TRUE/APPROX NEEDLE ATTENTIONS AND COSINE SIMILARITY---\n")
        line = (f"True attention on needle: {true_needle_attention.item():.4f}\n" 
                f"Approx attention on needle: {approx_needle_attention.item():.4f}\n"
                f"Output cosine similarity: {cosine.item():.4f}\n")
        f.write(line + "\n")

    print("True attention on needle:", true_needle_attention.item())
    print("Approx attention on needle:", approx_needle_attention.item())
    print("Output cosine similarity:", cosine.item())

    return A_true, A_approx, O_true, O_approx, true_needle_attention.item(),  approx_needle_attention.item(), cosine.item()

def get_true_attention_weights(query_head, key_head, value_head=None):
    """
    Compute the attention weights A for one attention head.

    query_head: [seq_len, head_dim]
    key_head:   [seq_len, head_dim]

    Returns:
        A: [seq_len, seq_len]
    """

    d = key_head.shape[-1]
    seq_len = query_head.shape[0]

    M = torch.triu(torch.full((seq_len, seq_len),float("-inf"), 
                              device=query_head.device, dtype=query_head.dtype,), diagonal=1,)

    scores = (query_head @ key_head.T) / math.sqrt(d)

    A = torch.softmax(M + scores, dim=-1)

    return A

def find_needle_heads(model, tokenizer, messages, needle, top_k=20, num_layers=None, num_heads=None):
    needle_positions = find_token_positions(tokenizer, messages, needle)

    if len(needle_positions) == 0:
        raise ValueError("Needle not found in tokenized input.")

    results = []

    n_layers = len(model.model.layers) if num_layers is None else num_layers
    n_heads = model.config.num_key_value_heads if num_heads is None else num_heads

    total_iters = n_layers * n_heads

    # Flattened loop with progress bar
    for idx in tqdm(range(total_iters), desc="Scanning heads"):
        layer_idx = idx // n_heads
        head_idx = idx % n_heads

        key_head, value_head, query_head = get_kvq(messages, layer_idx=layer_idx, head_idx=head_idx, want_print=False, model=model, tokenizer=tokenizer)

        A = get_true_attention_weights(key_head, query_head, value_head)

        q_idx = -1

        needle_attention = A[q_idx, needle_positions].sum().item()
        max_attention = A[q_idx].max().item()
        mean_attention = A[q_idx].mean().item()

        results.append({
            "layer": layer_idx,
            "head": head_idx,
            "needle_attention": needle_attention,
            "max_attention": max_attention,
            "mean_attention": mean_attention,})

    results = sorted(
        results,
        key=lambda x: x["needle_attention"],
        reverse=True,)

    output_path = "reports/tables/top_needle_heads.txt"

    with open(output_path, "a") as f:
        f.write("--- TOP NEEDLE HEADS ---\n")

        print("\n--- TOP NEEDLE HEADS ---")
        for r in results[:top_k]:
            line = (
                f"layer={r['layer']:2d}, head={r['head']:2d} | "
                f"needle_attn={r['needle_attention']:.6f} | "
                f"max_attn={r['max_attention']:.6f} | "
                f"mean_attn={r['mean_attention']:.6f}"
            )

            print(line)
            f.write(line + "\n")
        f.write("\n")

    return results

def get_random_messages(path, num_tokens,local_window_size=0):
    """
    Read a text file and return a random token window that contains the needle.

    The needle is placed at a random position inside the window.
    If the document is shorter than num_tokens, the document is not repeated.
    """

    text_full = Path(path).read_text(encoding="utf-8")
    tokenizer = _get_tokenizer()

    # Get page number
    needle_num = int(re.search(r"page_(\d+)", path).group(1))

    # Load needle
    needle_path = Path('/'.join((path.split('/')[:-2])) + '/needles.csv')
    with open(needle_path) as file:
        needle = file.read().splitlines()[needle_num - 1]

    # Load prompt
    prompt_path = Path('/'.join((path.split('/')[:-2])) + '/prompt_questions.txt')
    with open(prompt_path) as file:
        prompt = file.read().splitlines()[needle_num - 1]

    # Tokenize full document and needle
    token_ids = tokenizer(text_full, add_special_tokens=False)["input_ids"]
    needle_ids = tokenizer(needle, add_special_tokens=False)["input_ids"]

    if len(token_ids) == 0:
        raise ValueError("Document is empty.")

    # Find the needle in the tokenized document
    needle_start = None
    for i in range(len(token_ids) - len(needle_ids) + 1):
        if token_ids[i:i + len(needle_ids)] == needle_ids:
            needle_start = i
            break

    if needle_start is None:
        raise ValueError("Needle was not found in the tokenized document.")

    # Do not repeat short documents
    window_len = min(num_tokens, len(token_ids))

    if len(needle_ids) > window_len:
        raise ValueError("The selected window is too small to contain the full needle.")

    # Random position of the needle inside the selected window
    needle_offset_in_window = random.randint(0, window_len - len(needle_ids)-local_window_size)

    # Start the window so that the needle appears at the chosen random offset
    window_start = (needle_start - needle_offset_in_window) % len(token_ids)

    # Take window_len tokens, wrapping only if necessary
    selected_token_ids = [token_ids[(window_start + j) % len(token_ids)] for j in range(window_len)]
    text = tokenizer.decode(selected_token_ids, skip_special_tokens=True)

    # Messages 
    messages = [
    {"role": "system", "content": "You will recieve a question of the form 'What is the secret (key) in the document?' and must answer in the form 'The secret (key) is (value).'."}, # Besked til modellen om hvordan den skal opføre sig
    {"role": "user", "content": f"Read the following text and answer the question: '{prompt}' You must only use the provided information to answer.\nText:\n{text}"}, # Besked fra user (os)
    ]

    return messages, text, needle

"""__________OPTIONAL TEST______________"""
# messages, text, needle = get_random_messages("document-haystack/AIG/AIG_25Pages/Text_TextNeedles/AIG_25Pages_TextNeedles_page_4.txt" , num_tokens=200)
# print("_____MESSAGES______\n")
# print(messages)
# print("\n")
# print("_____TEXT______\n")
# print(text)
# print("\n")
# print("_____NEEDLE______\n")
# print(needle)
# print("\n")
