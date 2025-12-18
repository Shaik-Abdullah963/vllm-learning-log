# Key metrics for LLM Inference
## Phase 0 - Time To First Token
**TTFT (Time To First Token):** The time it takes to generate the first token after sending a request. It reflects how fast the model can start responding.
This metric captures:
- model loading and tokenization
- prompt prefill latency
- first decode step

TTFT was measured using streaming generation with a separate
generation thread and `TextIteratorStreamer`.

---

### Model Configuration

- **Model**: `distilgpt2`
- **Framework**: Hugging Face Transformers
- **Device**: MPS (Apple Silicon)
- **Max New Tokens**: 20


---


### Input Prompt

```text
An increasing sequence: one,
```

### Input Token Statistics

- **Batch size**: 1
- **Tokens per sequence**: 6
- **Total input tokens (including padding)**: 6
- **Total real input tokens (non-padding)**: 6

> Note: In this benchmark, batch size is 1 and no padding is applied,
> so all token counts are equal. These fields are listed separately
> to support future benchmarking with batching and variable-length inputs.

---

### Example Output
![TTFT Output](results/ttft.png)


### TTFT Results

TTFT was measured across multiple runs to distinguish between **cold-start**
and **warm-start** behavior.

- **Cold-start TTFT**: **~1.08 s**  
  The first run after process startup. This includes model initialization,
  device placement on MPS, kernel compilation, and memory setup.

- **Warm-start TTFT**: **~0.86–0.90 s**  
  Subsequent runs after initialization, where kernels and memory are reused
  and the model is already resident on the device.

> **Note:** Cold-start latency is expected to be higher and is relevant for
> scenarios such as server restarts and autoscaling. Warm-start TTFT reflects
> steady-state performance and is the primary metric for ongoing inference
> workloads.