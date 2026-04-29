# Load model directly
import math
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@lru_cache(maxsize=1)
def _get_tokenizer() -> AutoTokenizer:
    """Return a cached tokenizer instance for consistent token counting."""
    return AutoTokenizer.from_pretrained(MODEL_ID)

def load_model():
    """Load the LLM and tokenizer.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True)
    tokenizer = _get_tokenizer()

    print("-------------- MODEL DEVICE --------------")
    # Bare et tjek at den rent faktisk kører på GPU haha
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    print(model.device)

    return model, tokenizer

def get_messages(path, num_tokens):
    """Read a text file and get its last ``num_tokens`` tokens as text. Return the message template with the text and needle prompt inserted.

    Args:
        path: Path to a UTF-8 encoded text file.
        num_tokens: Number of trailing tokens to keep.

    Returns:
        Message template with the text and needle prompt inserted, the text, the needle
    """
    text = Path(path).read_text(encoding="utf-8")
    tokenizer = _get_tokenizer()
    # Tokenize text
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    # Get last num_tokens tokens
    last_token_ids = token_ids[-num_tokens:]
    text = tokenizer.decode(last_token_ids, skip_special_tokens=True)

    # Get the needle
    needle_num = int(path[-5])
  
    needle_path=Path('/'.join((path.split('/')[:-2]))+'/needles.csv')
    with open(needle_path) as file:
        needle=file.read().splitlines()[needle_num-1]

    # get the promt
    prompt_path=Path('/'.join((path.split('/')[:-2]))+'/prompt_questions.txt')
    with open(prompt_path) as file:
        prompt=file.read().splitlines()[needle_num-1]

    # Messages 
    messages = [
    {"role": "system", "content": "You will recieve a question of the form 'What is the secret (key) in the document?' and must answer in the form 'The secret (key) is (value).'."}, # Besked til modellen om hvordan den skal opføre sig
    {"role": "user", "content": f"Read the following text and answer the question: '{prompt}' You must only use the provided information to answer.\nText:\n{text}"}, # Besked fra user (os)
    ]

    #Return last num_tokens of text
    return messages, text, needle

def get_response(model, tokenizer, messages):
    """Generate a model response for a chat-style message list.

    Args:
        model: Loaded causal language model.
        tokenizer: Tokenizer associated with the model.
        messages: Chat messages in role/content format.

    Returns:
        tuple: Tokenized inputs, generation outputs object, and generated token ids.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)

    # Genererer output tokens (LLM'ens svar)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        eos_token_id=tokenizer.eos_token_id, # Indsæt stop-token
        pad_token_id=tokenizer.eos_token_id,
        # output_attentions=True,
        use_cache=True, 
        return_dict_in_generate=True,
    )
    generated_tokens = outputs.sequences[0]
    return inputs, outputs, generated_tokens

def extract_KV(outputs, layer_idx, head_idx):
    """Extract key/value cache tensors for a specific layer and attention head.

    Args:
        outputs: Generation outputs containing ``past_key_values``.
        layer_idx: Transformer layer index to inspect.
        head_idx: Attention head index to inspect.

    Returns:
        tuple: Full KV cache layers, full key tensor, full value tensor,
        selected key head tensor, and selected value head tensor.
    """
    KV_cache = outputs.past_key_values.layers
    layer = KV_cache[layer_idx]
    key = layer.keys # [batch_size, num_heads, seq_len, head_dim]
    value = layer.values # [batch_size, num_heads, seq_len, head_dim]

    key_head = key[0][head_idx] # [0] fordi, der er 1 batch
    value_head =  value[0][head_idx]
    return KV_cache, key, value, key_head, value_head


def extract_query(model, inputs, layer_idx, head_idx):
    """Extract query tensor and one query head for a given transformer layer.

    Args:
        model: Loaded causal language model.
        inputs: Tokenized model inputs.
        layer_idx: Transformer layer index.
        head_idx: Query head index.

    Returns:
        Tuple with full query tensor [batch, num_heads, seq_len, head_dim] and
        one query head [seq_len, head_dim] for batch index 0.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        forward_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    layer_module = model.model.layers[layer_idx]
    # hidden_states[i] is the input representation to transformer layer i.
    hidden_in = forward_outputs.hidden_states[layer_idx]
    normed_hidden = layer_module.input_layernorm(hidden_in)
    query = layer_module.self_attn.q_proj(normed_hidden)

    batch_size, seq_len, _ = query.shape
    num_heads = getattr(layer_module.self_attn, "num_heads", model.config.num_attention_heads)
    head_dim = getattr(layer_module.self_attn, "head_dim", query.shape[-1] // num_heads)
    # Reshape [B, T, H*D] -> [B, H, T, D] to index a specific attention head.
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    query_head = query[0, head_idx]
    return query, query_head

def get_kvq(messages, layer_idx, head_idx, want_print=False, model=None, tokenizer=None):
    # Load model if not given
    if model == None:
        model, tokenizer = load_model()
    elif tokenizer == None:
        tokenizer = _get_tokenizer

    # Generate response
    inputs, outputs, generated_tokens = get_response(model, tokenizer, messages)

    # KV cache
    KV_cache, key, value, key_head, value_head = extract_KV(outputs, layer_idx=layer_idx, head_idx=head_idx)

    # Get query matrix
    query_inputs = {
        "input_ids": generated_tokens.unsqueeze(0),
        "attention_mask": torch.ones_like(generated_tokens).unsqueeze(0),
    }
    query, query_head = extract_query(model, query_inputs, layer_idx=layer_idx, head_idx=head_idx)

    # Resize Q to T-1:
    kv_seq_len = key.shape[2]
    if query.shape[2] != kv_seq_len:
        query = query[:, :, :kv_seq_len, :]
        query_head = query_head[:kv_seq_len, :]

    # Error if not float32
    query_head = query_head.to(torch.float32)
    key_head = key_head.to(torch.float32)
    value_head = value_head.to(torch.float32)

    if want_print:
        print("-------------- SYSTEM PROMPT AND REPLY --------------")
        input_length = inputs.input_ids.shape[1]
        print("Input length:", input_length)

        print(tokenizer.decode(generated_tokens, skip_special_tokens=True))

        print("-------------- REPLY ONLY --------------")
        print(tokenizer.decode(generated_tokens[input_length:], skip_special_tokens=True), "\n")

        print(f"{len(KV_cache)} transformer layers in the model")

        print(f"KV cache from transformer layer {layer_idx}:")
        print(f"K dimension: {key.shape}")
        print(f"V dimension: {value.shape}\n")

        print(f"KV cache from transformer layer {layer_idx} and head {head_idx}:")
        print(f"Key dimension: {key_head.shape}")
        print(f"Value dimension: {value_head.shape}")
        print(f"Query dimension: {query_head.shape}\n")
    return key_head, value_head, query_head

def get_true_attention_values(query_head, key_head, value_head):
    M = torch.triu(torch.full((query_head.shape[0], query_head.shape[0]), float("-inf"), device=query_head.device, dtype=query_head.dtype),diagonal=1)
    attn_values = torch.softmax((M + (query_head @ key_head.T)), dim=-1) @ value_head
    return attn_values