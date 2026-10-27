import time
import numpy as np
from llm_sdk import Small_LLM_Model
import json

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IM_END_ID = 151645

def generate(model: Small_LLM_Model, prompt: str, max_tokens: int) -> str:
    encoded = model.encode(prompt)[0].tolist()

    prompt_len = len(encoded)
    for _ in range(max_tokens):
        # print(f"Encoded: {encoded}")
        logits: list[float] = model.get_logits_from_input_ids(encoded)
        max_id = logits.index(max(logits))
        # print(f"Token ID: {max_id}\nScore :{max(logits)}\nToken: {model.decode(max_id)}")
        if max_id == IM_END_ID:
            break
        encoded.append(max_id)
    # completed_prompt = model.decode(encoded)
    answer = model.decode(encoded[prompt_len:])
    # print(f"Completed prompt:\n{completed_prompt}")
    return answer
    # print(len(logits))


def time_one_step(model: Small_LLM_Model, ids: list[int]) -> float:
    """Renvoie la durée en secondes d'un appel à get_logits_from_input_ids."""
    start = time.perf_counter()
    model.get_logits_from_input_ids(ids)
    elapsed = time.perf_counter() - start

    return elapsed


def init_llm():
    start = time.perf_counter()
    model = Small_LLM_Model()

    elapsed = time.perf_counter() - start

    import torch
    print(f"Mémoire GPU allouée : {torch.cuda.memory_allocated() / 1e9:.2f} Go")

    print(f"Time to instantiate model : {elapsed:.1f}")

    question = 'Answer only with JSON like {"name": ..., "parameters": {...}}. What is the sum of 2 and 3?'
    prompt = f'{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n'
    prompt_wothink = f'{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n<think>\n\n</think>\n\n'

    short_prompt = "What is the sum of 2 and 3? " * 2
    long_prompt = "What is the sum of 2 and 3? " * 40

    short_encoded = model.encode(short_prompt)[0].tolist()
    long_encoded = model.encode(long_prompt)[0].tolist()

    time_one_step(model, short_encoded)

    short_times = [time_one_step(model, short_encoded) for _ in range(5)]
    print(len(short_encoded))
    print(short_times)
    long_times = [time_one_step(model, long_encoded) for _ in range(5)]
    print(len(long_encoded))
    print(long_times)
    # max_tokens = 50
    # answer1 = generate(model, question, max_tokens)
    # answer2 = generate(model, prompt, max_tokens)
    # answer3 = generate(model, prompt_wothink, max_tokens)    
    # print(answer1)
    # print("\n")
    # print(answer2)
    # print("\n")
    # print(answer3)

    


if __name__ == "__main__":
    init_llm()
