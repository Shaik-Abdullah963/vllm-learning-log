# Key metrics for LLM Inference
## Phase 0 - Time To First Token
**TTFT (Time To First Token):** The time it takes to generate the first token after sending a request. It reflects how fast the model can start responding.
This metric captures:
- model loading and tokenization
- prompt prefill latency
- first decode step

TTFT was measured using streaming generation with a separate
generation thread and `TextIteratorStreamer`.

### Example Output
![TTFT Output](results/ttft.png)