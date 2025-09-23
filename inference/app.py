import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch 
import streamlit as st
from model.model import GPT,GPTConfig
import tiktoken
device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")
cfg = GPTConfig(
    vocab_size=50257,    
    block_size=128,     
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)
model = GPT(cfg).to(device)
model.load_state_dict(torch.load(
    r'C:\Users\PARTHA SARATHI\Python\slm-model\saved_models\best_model_params.pt',
    map_location=device
))
model.eval()
st.title("MiniMind 🧠")
st.write("Enter a prompt and generate text with your trained GPT model.")
prompt = st.text_input("Prompt:", "Once upon a time")
max_new_tokens = st.slider("Max new tokens", 10, 200, 50)
temperature = st.slider("Temperature", 0.1, 2.0, 1.0)
top_k = st.slider("Top-k (0 = disabled)", 0, 500, 50)
if st.button("Generate"):
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None
        )
        output_text = enc.decode(output_ids[0].tolist())
    st.subheader("Generated Text")
    st.write(output_text)