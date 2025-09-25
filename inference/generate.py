import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import click
from model.model import GPT,GPTConfig
import torch
import tiktoken
cfg = GPTConfig(
    vocab_size=50257,    
    block_size=128,     
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)
model=GPT(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
@click.command()
@click.option("--prompt",prompt="enter your prompt",help="here enter your prompt")
@click.option("--mt",prompt="enter max tokens",help="max new tokens",default=100)
@click.option("--temp",help="temperature",default=0.8)
@click.option("--tk",help="top_k values",default=200)
def generate(prompt,mt,temp,tk):
    # WORKS - relative path
    model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'best_model_params.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    tokenizer=tiktoken.get_encoding("gpt2")
    input_ids=torch.tensor([tokenizer.encode(prompt)],dtype=torch.long).to("cuda"if torch.cuda.is_available() else "cpu") 
    with torch.no_grad():
           output_ids=model.generate(
        input_ids,
        max_new_tokens=mt,
        temperature=temp,
        top_k=tk if tk>0 else None
    )
    output_Text=tokenizer.decode(output_ids[0].tolist())
    click.echo(f"Generated Text: {output_Text}")
if __name__ == "__main__":
      generate()