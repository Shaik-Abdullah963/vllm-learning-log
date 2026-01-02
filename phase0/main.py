import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

def main():
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype="auto", device_map="auto")
    inputs = tok(["An increasing sequence: one,"], return_tensors="pt").to(model.device)
    print(inputs)
    print("Tokens per sequence: ", inputs["input_ids"].shape[1])
    print("Batch size: ", inputs["input_ids"].shape[0])
    print("Input length (Total tokenns including padding): ",inputs["input_ids"].numel())
    print("Total real tokens: ", (inputs["attention_mask"]).sum().item())
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    t0 = time.perf_counter()
    thread.start()
    first_token_time = None
    generated_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = time.perf_counter()
            # Time to first token
            print(f"\n TTFT: {first_token_time - t0:.4f}s\n")
        last_token_time = now
        generated_text += new_text
    # Total Latency or End to End Latency(E2EL)
    print(f"\n E2EL: {last_token_time - t0:.4f}s\n")
    # Token Generation Time (TGT)
    # Token Generation Time measures only the decode phase, i.e. all tokens after the first one.
    token_generation_time = last_token_time - first_token_time
    print(f"Token Generation Time: {token_generation_time:.4f}s")
    print(generated_text)
    thread.join()

if __name__ == "__main__":
    main()