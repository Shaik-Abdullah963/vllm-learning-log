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
    temperature: float = 1.0
    top_p: float = 0.9


#Token Smapler - It takes the model's raw output scores for the next token and decides which token ID to pick it.
def sample_next_token(logits, temperature: float, top_p: float):
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

@torch.inference_mode()
def generate_full_text(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    

    t0 = time.perf_counter()

    # ----------- PREFILL-----------
    out = model(**inputs, use_cache=True)
    past = out.past_key_values

    # first token selection time = TTFT boundary (prefill + first decode step compute)
    #We measure time when we *finish* the first next token
    next_token = sample_next_token(out.logits[:, -1, :], temperature, top_p)
    t_first = time.perf_counter()
    
    generated_ids = [next_token]
    itl_times = [] # timestamps of each decode step completion after the first token
    last_token = next_token

    # ----------- DECODE (token-by-token) -----------
    #Autoregressive decoding loop, generates one token at a time using KV-cache
    for _ in range(max_new_tokens -1):
        # Decode step
        out = model(input_ids = last_token, past_key_values = past, use_cache=True)

        #Save the updated KV cache, andthis cache now includes the newly generated token, and will be used in next iteration
        #O(1) operation
        past = out.past_key_values

        # Sample next token
        last_token = sample_next_token(out.logits[:, -1, :], temperature, top_p)
        generated_ids.append(last_token)

        # Inter token latency time capture
        itl_times.append(time.perf_counter())
    t_end = time.perf_counter()

    gen_ids = torch.cat(generated_ids, dim=1)  #[B, max_new_tokens]

    #decode tokens to text
    gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)

    #metrics
    ttft = t_first - t0
    e2el = t_end - t0

    output_tokens = gen_ids.shape[1]

    # ITLs: exact pauses between consecutive tokens in decode
    # We have completion times for each token computation.
    # token 1 completes at t_first, token 2 completes at itl_times[0], etc.

    token_finish_times = [t_first] + itl_times  #len = max_new_tokens
    itls = [
        token_finish_times[i] - token_finish_times[i - 1]
        for i in range(1, len(token_finish_times))
    ]

    # TPOT (single request) = (E2EL - TTFT)/(output_tokens-1)  (decode avg)
    tpot = (e2el - ttft) / (output_tokens - 1) if output_tokens > 1 else 0.0

    # mean ITL (single request) equals TPOT by definition when computed on same tokens
    mean_itl = (sum(itls) / len(itls)) if itls else 0.0

    return {
        "text": prompt + gen_text,
        "ttft_s": ttft,
        "e2el_s": e2el,
        "output_tokens": int(output_tokens),
        "tpot_s": tpot,
        "mean_itl_s": mean_itl,
        "itls_s": itls,  
    }


@app.get("/")
def health():
    return {"ok": True, "model": MODEL, "device": str(model.device)}

@app.post("/generate")
def generate(req: Req):
    """
    Non-streaming: returns full text at the end.
    TTFT here is “compute TTFT” (prefill+first token compute), not “network TTFT”.
    """
    res = generate_full_text(req.prompt, req.max_new_tokens, req.temperature, req.top_p)

    # Often you won’t return itls_s in /generate; you log it.
    res.pop("itls_s", None)
    return res

@app.post("/generate/stream")
@torch.inference_mode()
async def generate_stream(req: Req):
    """
    Streaming SSE: emits one token at a time + meta + summary.
    This gives “user-perceived TTFT” because client receives first token immediately.

    """
    prompt = req.prompt
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # Asynce generator that streams SSE events
    async def event_gen():
        t0 = time.perf_counter()
        
        # ----------- PREFILL-----------#
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values

        # First Token generation and TTFT
        next_token = sample_next_token(out.logits[:, -1, :], req.temperature, req.top_p)
        t_first = time.perf_counter()
        ttft = t_first - t0

        #Convert first token to text and emit "meta" + first token
        first_text = tok.decode(next_token[0], skip_special_tokens=True)
        yield {"event": "meta", "data": f'{{"ttft_s": {ttft:.6f}}}'}
        yield {"event": "token", "data": first_text}
        output_tokens = 1
        token_finish_times = [t_first]  # finish time for token 1
        last_token = next_token

        # ----------- DECODE (token-by-token) -----------
        for _ in range(req.max_new_tokens - 1):
            #Decode step
            out = model(input_ids=last_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            last_token = sample_next_token(out.logits[:, -1, :], req.temperature, req.top_p)

            #inter token latency time capture
            t_now = time.perf_counter()
            token_finish_times.append(t_now)
            token_text = tok.decode(last_token[0], skip_special_tokens=True)
            output_tokens += 1
            yield {"event": "token", "data": token_text}

        #E2EL time capture
        t_end = time.perf_counter()
        e2el = t_end - t0

        itls = [
                token_finish_times[i] - token_finish_times[i - 1]
                for i in range(1, len(token_finish_times))
            ]
        mean_itl = (sum(itls) / len(itls)) if itls else 0.0
        tpot = (e2el - ttft) / (output_tokens - 1) if output_tokens > 1 else 0.0

        yield {
                "event": "summary",
                "data": (
                    f'{{"e2el_s": {e2el:.6f}, "output_tokens": {output_tokens}, '
                    f'"tpot_s": {tpot:.6f}, "mean_itl_s": {mean_itl:.6f}}}'
                ),
            }

    return EventSourceResponse(event_gen())







            