import time
from llm_sdk import Small_LLM_Model


def init_llm():
    start = time.perf_counter()
    model = Small_LLM_Model()
    elapsed = time.perf_counter() - start

    import torch
    print(f"Mémoire GPU allouée : {torch.cuda.memory_allocated() / 1e9:.2f} Go")

    print(f"Time to instantiate model : {elapsed:.1f}")
    print(type(model))


if __name__ == "__main__":
    init_llm()
