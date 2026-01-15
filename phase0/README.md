
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
  {"text":"An increasing sequence: one, two, three. In China, it’s associated with the slowing of migration and another by",
  "ttft_s":0.08361095299187582,
  "e2el_s":0.6445639790035784,"output_tokens":20,
  "tpot_s":0.02952384347430014,
  "mean_itl_s":0.029523782894677043
  }

  Input:
  curl -N -X POST http://localhost:8000/generate/stream \  
  -H "Content-Type: application/json" \
  -d '{"prompt":"An increasing sequence: one,", "max_new_tokens":20}'

  Output:
  event: meta
  data: {"ttft_s": 0.129779}

  event: token
  data:  two

  event: token
  data: ,

  event: token
  data:  and

  event: token
  data:  four

  event: token
  data: ,

  event: token
  data:  all

  event: token
  data:  ruled

  event: token
  data:  by

  event: token
  data:  animals

  event: token
  data: .

  event: token
  data:  Notably

  event: token
  data: ,

  event: token
  data:  his

  event: token
  data:  sem

  event: token
  data: ip

  event: token
  data: rot

  event: token
  data: ective

  event: token
  data:  effect

  event: token
  data:  of

  event: token
  data:  both

  event: summary
  data: {"e2el_s": 0.730274, "output_tokens": 20, "tpot_s": 0.031605, "mean_itl_s": 0.031585}



```

Purpose:
Establish prefill vs decode latency and KV-cache behavior before comparing with vLLM.

