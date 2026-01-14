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

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 20
    temperature: float = 0.8
    top_p: float = 0.95

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "device": str(model.device)}