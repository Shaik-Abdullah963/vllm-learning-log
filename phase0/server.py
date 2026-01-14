import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sse_starlette import EventSourceResponse

MODEL = "distilgpt2"
app = FastAPI(title = "HF distilgpt2 Inference Metrics Demo")

#Load once
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto")
model.eval()


#Token Smapler - It takes the model's raw output scores for the nexxt token and decides which token ID to pick it.
def smaple_next_token(logits, temperature: float, top_p: float):
    """
    logits: [B,V]
    returns: next_token_ids [B, 1]

    """
    if temperature <= 0:
        # Greedy decoding
        next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token_ids
    # Apply temperature
    scalled_logits = logits / temperature

    # Apply Softmax to get probabilities
    probs = torch.softmax(scalled_logits, dim=-1)

    # Apply Top-p (nucleus) sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens with cumulative probability above top_p
    mask = cumulative_probs > top_p

    #Sometimes the top token alone already exceeds top_p, so everything becomes masked
    # In that case, we keep at least one token
    mask[...,0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)

    #Renormalize the probabilities
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # sample in sorted space then map back
    next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_in_sorted)
    return next_token


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "device": str(model.device)}