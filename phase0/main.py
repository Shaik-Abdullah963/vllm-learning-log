import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

def main():
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", dtype="auto", device_map="auto")
    model.eval()
    inputs = tok(["An increasing sequence: one,"], return_tensors="pt").to(model.device)
    print(inputs)
    print("Tokens per sequence: ", inputs["input_ids"].shape[1])
    print("Batch size: ", inputs["input_ids"].shape[0])
    print("Input length (Total tokenns including padding): ",inputs["input_ids"].numel())
    total_real_tokens = (inputs["attention_mask"]).sum().item()
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5)

    print("Total real tokens: ", (inputs["attention_mask"]).sum().item())
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
    t0 = time.perf_counter()
    def _run():
        with torch.inference_mode():
            model.generate(**generation_kwargs)
    thread = Thread(target=_run)
    
    thread.start()
    first_token_time = None
    last_token_time = None
    generated_text = ""
    token_times = []
    for new_text in streamer:
        
        now = time.perf_counter()
        token_times.append(now)
        if first_token_time is None:
            first_token_time = now
            # Time to first token
            ttft= first_token_time - t0
            print(f"\n TTFT: {ttft:.4f}s\n")
        last_token_time = now
        generated_text += new_text
    # # Total Latency or End to End Latency(E2EL)
    e2el = last_token_time - t0
    print(f"\n E2EL: {e2el:.4f}s\n")
    # Token Generation Time (TGT)
    # Token Generation Time measures only the decode phase, i.e. all tokens after the first one.
    token_generation_time = last_token_time - first_token_time
    print(f"Token Generation Time: {token_generation_time:.4f}s")
    print(generated_text)

    #Other metrics
    total_output_tokens = len(token_times)
    time_per_output_token = token_generation_time / (total_output_tokens-1)
    print(f"Time per output token: {time_per_output_token:.4f}s")

    #Inter toke Latency : for each token after the first one, how much time it took to generate it on average
    #for single request inter token latency is same TPOT
    print(f"Inter Token Latency: {time_per_output_token:.4f}s")



    thread.join()

if __name__ == "__main__":
    main()