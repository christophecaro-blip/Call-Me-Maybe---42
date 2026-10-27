import time
import numpy as np
from llm_sdk import Small_LLM_Model
import json


def init_llm():
    model = Small_LLM_Model()

    path_vocab = model.get_path_to_vocab_file()
    path_tokeniser = model.get_path_to_tokenizer_file()
    with open(path_tokeniser, "r", encoding="utf-8") as ft:
        tok = json.load(ft)
    print(f"Keys: {tok.keys()}")
    print(f"Len of added tokens: {len(tok['added_tokens'])}")
    print(f"First Added tokens : {tok['added_tokens'][0]}")
    print(f"Last Added tokens : {tok['added_tokens'][-1]}")
    with open(path_vocab, "r", encoding="utf-8") as fv:
        vocab = json.load(fv)
    print(f"Len of vocab and max values: {len(vocab), max(vocab.values())}")

    vocab_ids = set(vocab.values())
    vocab_texts = set(vocab.keys())

    for entry in tok["added_tokens"]:
        token_id = entry["id"]
        content = entry["content"]
        special = entry["special"]
        print(f"{token_id:>7}  {content!r:25}  "
            f"special : {special!s:5}  "
            f"id dans vocab: {token_id in vocab_ids}  "
            f"texte dans vocab: {content in vocab_texts}")

    mixed = [t for t in vocab if '"' in t]
    print(f"Len of mixed:{len(mixed)}")
    for c in [",", "}", ":"]:
        found = sorted([t for t in mixed if c in t], key=len)
        print(c, len(found), found[:12])



if __name__ == "__main__":
    init_llm()
