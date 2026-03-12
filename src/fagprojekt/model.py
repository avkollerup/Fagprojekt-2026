# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

model_id = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("-------------- MODEL DEVICE --------------")
# Bare et tjek at den rent faktisk kører på GPU haha
print(torch.cuda.get_device_name(0))
print(model.device)

messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",}, # Besked til modellen om hvordan den skal opføre sig
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"}, # Besked fra user (os)
 ]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
).to(model.device)

# Genererer output tokens (LLM'ens svar)
outputs = model.generate(
    **inputs, 
    max_new_tokens=100, 
    eos_token_id=tokenizer.eos_token_id, # Indsæt stop-token
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True, 
    return_dict_in_generate=True,
)

# outputs.sequences contains prompts plus generated tokens
# Print system prompt and reply
generated_tokens = outputs.sequences[0]
print("-------------- SYSTEM PROMPT AND REPLY --------------")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))

# Print only reply
input_length = inputs.input_ids.shape[1]
print("-------------- REPLY ONLY --------------")
print(tokenizer.decode(generated_tokens[input_length:], skip_special_tokens=True))

# KV CACHE
print("-------------- KV CACHE --------------")
KV_cache = outputs.past_key_values.layers
print(len(KV_cache)) # 32 tranformer layers i modellen

# Extract KV cache from specific transformer layer:
# Fra https://huggingface.co/docs/transformers/v5.3.0/en/internal/generation_utils#transformers.DynamicCache
# transformers.DynamicCache stores the key and value states as a list of CacheLayer, one for each layer. 
layer_idx = 0
layer = KV_cache[0]
key = layer.keys
value = layer.values

print(f"KV cache from transformer layer {layer_idx}")
print(f"K dimension: {key.shape}")
print(f"V dimension: {value.shape}")

# How to get KV cache from specific head in a specific layer???
# Fra https://huggingface.co/docs/transformers/v5.3.0/en/internal/generation_utils#transformers.DynamicCache
# The expected shape for each tensor in the CacheLayers is [batch_size, num_heads, seq_len, head_dim]
# I guess udtrække fra en specifik head_idx.



# Indsætte KV cachen igen til næste prompt:
# Der er et argument, der hedder past_key_values!!
# outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)



