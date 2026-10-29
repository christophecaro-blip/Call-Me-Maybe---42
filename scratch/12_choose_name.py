from pathlib import Path

import numpy as np

from src.decoding.constraints import ChoiceConstraint
from src.decoding.decoder import decode_value
from src.io_utils import load_functions, load_prompts
from src.llm.engine import Engine
from src.prompting import build_prompt

functions = load_functions(Path("data/input/functions_definition.json"))
prompts = load_prompts(Path("data/input/function_calling_tests.json"))
names = [fn.name for fn in functions]
engine = Engine()


def top_allowed(logits, constraint, k=5):
    """Return the k best (text, logit) pairs the constraint accepts."""
    found = []
    for token_id in np.argsort(logits)[::-1]:
        text = engine.vocab.id_to_text[token_id]
        if text is not None and constraint.accepts(text):
            found.append((text, float(logits[token_id])))
            if len(found) == k:
                break
    return found


for item in prompts:
    prompt_text = build_prompt(functions, item.prompt)
    prompt_ids = engine.encode(prompt_text)
    ids = prompt_ids + engine.encode('{"name": "')
    logits = engine.next_logits(ids)
    top = top_allowed(logits, ChoiceConstraint(options=names))
    _, text = decode_value(engine, ids, ChoiceConstraint(options=names), 20)
    print(f"{text[:-1]:35} ← {item.prompt}")  # [:-1] enlève le " fermant
    for text_k, score in top:
        print(f"    {text_k!r:15} {score:.2f}")
