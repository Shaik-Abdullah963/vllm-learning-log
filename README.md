# vLLM Inference Learning Log 🚀  
*A systems-first journey into LLM inference performance*

---

## Why this repository exists

This repository documents my **hands-on learning journey into Large Language Model (LLM) inference and performance optimization**, with a specific focus on **understanding vLLM from first principles**.

Instead of treating vLLM as a black-box library or attempting to build a full “vLLM clone” upfront, this repo follows a **systems engineering approach**:

- isolate one inference mechanism at a time  
- run small, focused experiments  
- measure real performance behavior  
- build intuition before reading complex code  

This is a **learning log**, not a tutorial or production framework.

---

## Core philosophy

### ❌ Wrong way to learn vLLM
- “Build one big inference gateway”
- “Read the entire vLLM codebase”
- “Deploy vLLM and tweak flags randomly”

### ✅ Right way to learn vLLM
- Treat vLLM as **multiple interacting subsystems**
- Learn **one mechanism at a time**
- Validate understanding using **measurements**
- Connect the dots **only after intuition is built**

This mirrors how vLLM itself is designed.

---

## The correct mental model

vLLM is not a single component — it is a **stack of systems**:

- API & serving layer  
- Runtime scheduler (continuous batching)  
- Memory systems (KV cache, paging)  
- Model execution (attention, decoding, kernels)  
- Distributed runtime  

Understanding everything at once leads to confusion.  
This repository progresses **vertically through the stack**, not horizontally.

---

## The 5-Layer vLLM learning stack

You do **not** start at the top.

```text
Layer 5: Distributed inference (multi-GPU, multi-node)
Layer 4: Runtime scheduling (continuous batching, prefill vs decode)
Layer 3: Memory systems (KV cache, PagedAttention)
Layer 2: Model execution (attention, decoding, kernels)
Layer 1: API & serving (OpenAI-compatible server)
