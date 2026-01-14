import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL = "distilgpt2"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", dtype="auto")
    model.eval()

    prompt = "An increasing sequence: one,"
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    print(inputs)
    print("Prompt tokens:", inputs["input_ids"].shape[1])
    print("Batch size:", inputs["input_ids"].shape[0])
    print("Total real tokens:", inputs["attention_mask"].sum().item())

    # Warmup (avoids first-run overhead skew)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=20)

    t0 = time.perf_counter()

    def _run():
        with torch.inference_mode():
            model.generate(**generation_kwargs)

    thread = Thread(target=_run)
    thread.start()

    first_chunk_time = None
    last_time = None
    generated_text = ""
    chunk_times = []

    for chunk in streamer:
        now = time.perf_counter()
        if first_chunk_time is None:
            first_chunk_time = now
            print(f"\nTTFT: {first_chunk_time - t0:.4f}s\n")

        print(chunk, end="", flush=True)
        generated_text += chunk
        chunk_times.append(now)
        last_time = now

    thread.join()

    e2el = last_time - t0 if last_time else 0.0
    decode_time = (last_time - first_chunk_time) if (first_chunk_time and last_time) else 0.0

    # Estimate output tokens by re-tokenizing the generated text
    out_tokens = len(tok(generated_text, return_tensors="pt")["input_ids"][0])

    decode_tps = (out_tokens / decode_time) if decode_time > 0 else 0.0

    print(f"\n\nE2EL: {e2el:.4f}s")
    print(f"Output tokens (estimated): {out_tokens}")
    print(f"Decode time: {decode_time:.4f}s")
    print(f"Decode TPS (estimated): {decode_tps:.2f} tokens/s")

if __name__ == "__main__":
    main()
