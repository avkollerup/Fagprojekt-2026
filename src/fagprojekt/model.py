# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

model_id = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "What is 2+2?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs, 
    max_new_tokens=50, 
    use_cache=True, 
    return_dict_in_generate=True,
)

# outputs.sequences contains prompts plus generated tokens
# Print prompt and newest reply
print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))

past_key_values = outputs.past_key_values
generated_tokens = outputs.sequences

print(past_key_values)
print(type(past_key_values), len(past_key_values))

