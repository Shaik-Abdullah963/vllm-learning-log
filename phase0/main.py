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

    # after model + inputs created
    def run_once(measure: bool):
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)

        first_token_time = None
        t0 = time.perf_counter()

        def _run():
            with torch.inference_mode():
                model.generate(**generation_kwargs)

        thread = Thread(target=_run)
        thread.start()

        for _ in streamer:
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
                if measure:
                    print(f"TTFT: {first_token_time - t0:.4f}s")
                break  # we only need first token for TTFT

        thread.join()

    # ---- Cold (measured) ----
    print("Cold run:")
    run_once(measure=True)

    # ---- Warm-up (not measured) ----
    for _ in range(2):
        run_once(measure=False)

    # ---- Warm (measured) ----
    print("Warm run:")
    run_once(measure=True)
if __name__ == "__main__":
    main()
