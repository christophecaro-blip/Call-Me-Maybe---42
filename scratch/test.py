import time
from llm_sdk import Small_LLM_Model


def init_llm():
    start = time.perf_counter()
    model = Small_LLM_Model()

    import torch
    print(f"Mémoire GPU allouée : {torch.cuda.memory_allocated() / 1e9:.2f} Go")

    elapsed = time.perf_counter() - start

    print(f"Time to instantiate model : {elapsed:.1f}")

    encoded = model.encode("What is the sum of 4 and 5 ?")[0].tolist()
    for _ in range(12):
        print(f"Encoded: {encoded}")
        logits: list[float] = model.get_logits_from_input_ids(encoded)
        max_id = logits.index(max(logits))
        print(f"Token ID: {max_id}\nScore :{max(logits)}\nToken: {model.decode(max_id)}")
        encoded.append(max_id)

    completed_prompt = [model.decode(word) for word in encoded]
    print(f"{completed_prompt}")


if __name__ == "__main__":
    init_llm()