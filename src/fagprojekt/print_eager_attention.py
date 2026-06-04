import inspect
import transformers.models.llama.modeling_llama as llama_modeling
from fagprojekt.model import load_model


"""Efter lidt research, er det umiddelbart i funktionen eager_attention_forward hvor attention bliver beregnet,
og derfor der hvor Hokus_pokus skal indsættes hvis det er det vi gerne vil"""

def see_llama_attention_function():
    model, tokenizer = load_model(want_print=False)

    print("Attention implementation used by config:")
    print(model.config._attn_implementation)

    print("\nSource file:")
    print(inspect.getfile(llama_modeling.eager_attention_forward))

    print("\neager_attention_forward signature:")
    print(inspect.signature(llama_modeling.eager_attention_forward))

    print("\neager_attention_forward source:")
    print(inspect.getsource(llama_modeling.eager_attention_forward))


if __name__ == "__main__":
    see_llama_attention_function()
