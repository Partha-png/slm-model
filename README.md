# Small Language Model (GPT-Style) From Scratch

This project implements a GPT-style small language model in **PyTorch**, trained on the **TinyStories** dataset, to demonstrate a full end-to-end pipeline for transformer-based language models.  
The work highlights model architecture design, dataset preprocessing, efficient training, evaluation, and inference deployment.

---

## Motivation

Large Language Models (LLMs) such as GPT have achieved state-of-the-art results across NLP tasks, but reproducing their functionality at scale is computationally expensive.  
This project aims to:

- Recreate a mini-GPT from scratch to understand transformer internals.  
- Train on TinyStories for interpretable, lightweight experiments.  
- Build a modular, research-style codebase for extensibility.  
- Provide evaluation benchmarks (loss, perplexity) and a Streamlit interface for interaction.  

---

## Model Architecture

- **Embedding Layer:** Token + positional embeddings  
- **Transformer Blocks:**  
  - Multi-Head Causal Self-Attention  
  - LayerNorm + Residual Connections  
  - MLP with GELU activation  
- **Decoder Head:** Linear projection tied with embeddings  

**Training Optimizations:**  
- Gradient accumulation  
- Mixed precision (AMP)  
- Cosine learning rate scheduler  

---

## Training Results

<img width="672" height="531" alt="Training Curve" src="https://github.com/user-attachments/assets/1482e35d-b18a-4b04-8ccb-e4840542696d" />

Both training and validation loss decrease smoothly, showing stable convergence.  

**Final Evaluation Metrics:**  
- Validation Loss: `2.36`  
- Validation Perplexity (PPL): `10.6`  

**Interpretation:**  
Random guessing across a 50k vocabulary yields PPL ≈ 50,000.  
Achieving 10.6 PPL indicates the model effectively captures linguistic structure despite its compact size.  

---

## Sample Generations

![Sample Output](https://github.com/user-attachments/assets/1285878f-de82-43e1-9b56-876a039e4857)

---

## Project Structure

```text
slm-model/
│── data/
│   ├── datapreprocessing.py     # Dataset tokenization + .bin serialization
│
│── model/
│   ├── model.py                 # Transformer (GPT) implementation
│   ├── config.py                # Model hyperparameters (GPTConfig)
│
│── training/
│   ├── train.py                 # Training loop
│   ├── utils.py                 # Loss estimation, helpers
│
│── inference/
│   ├── app.py                   # Streamlit interface
│   ├── generate.py              # CLI text generation
│
│── saved_models/
│   ├── best_model_params.pt     # Best validation checkpoint
│   ├── final_model.pt           # Final model after training
