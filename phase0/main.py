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
        if first_token_time is None:
            first_token_time = time.perf_counter()
            print(f"\n TTFT: {first_token_time - t0:.4f}s\n")
        generated_text += new_text
    print(generated_text)
    thread.join()

if __name__ == "__main__":
    main()