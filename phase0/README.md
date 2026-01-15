
### Logits Shape and Next-Token Selection

The transformer outputs logits for every position in the input sequence, so the raw shape is:

\[
\text{logits} \in \mathbb{R}^{B \times T \times V}
\]

where:

- \( B \) = batch size  
- \( T \) = sequence length (number of input tokens)  
- \( V \) = vocabulary size  

For next-token generation, we only use the logits of the last time step:

```python
next_token_logits = logits[:, -1, :]

```

## Phase 0 – HuggingFace Inference Baseline

Endpoints:
- POST /generate        -> full text + TTFT, E2EL, TPOT, mean ITL
- POST /generate/stream -> SSE streaming + user-perceived TTFT

Run:
uvicorn phase0.server:app --reload

```CLI
   Sample Input:
   curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"An increasing sequence: one,", "max_new_tokens":20}'

  Output:
  {
    "text":"An increasing sequence: one, two, and three, called eyewear as the C6\n\n\n\nA marking on the",
    "ttft_s":0.038806516997283325,
    "e2el_s":0.5960898729972541,
    "output_tokens":20,
    "tpot_s":0.029330702947366883,
    "mean_itl_s":0.029330629104821895
  }

  Input:
  curl -N -X POST http://localhost:8000/generate/stream \  
    -H "Content-Type: application/json" \
    -d '{"prompt":"An increasing sequence: one,", "max_new_tokens":20}'

  Output:
  event: meta
data: {"ttft_s": 0.040714}

event: token
data:  2

event: token
data: ,

event: token
data:  3

event: token
data: .

event: token
data: 
data: 

event: token
data: 
data: 

event: token
data: Now

event: token
data:  we

event: token
data: 're

event: token
data:  adding

event: token
data:  increased

event: token
data:  voltage

event: token
data:  to

event: token
data:  the

event: token
data:  parameter

event: token
data:  range

event: token
data: :

event: token
data: 
data: 

event: token
data: 0

event: token
data: .

event: summary
data: {"e2el_s": 0.639951, "output_tokens": 20, "tpot_s": 0.031539, "mean_itl_s": 0.031520}



```

Purpose:
Establish prefill vs decode latency and KV-cache behavior before comparing with vLLM.

