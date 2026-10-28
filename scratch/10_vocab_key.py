from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.llm.vocabulary import Vocabulary

SENTENCES = [
    "What is the sum of 2 and 3?",
    '{"name": "fn_greet", "parameters": {"name": "shrek"}}',
    "Hello\nworld",
    "Multiply 3.14 by -2.5",
    "What is 123456 + 789?",
    r"Replace all \d+ with NUMBERS",
    "Greet Zoé",
    "a  b   c",
    "",
    '{"x": 1e-5, "ok": true}',
    "I love 🦀",
    "漢字𝔸",
]

model = Small_LLM_Model()
vocab = Vocabulary(Path(model.get_path_to_vocab_file()), 151936)

for s in SENTENCES:
    ids = model.encode(s)[0].tolist()
    print(s, [(i, vocab.id_to_text[i]) for i in ids])
    rebuilt = "".join(vocab.id_to_text[i] for i in ids)            # le texte de chaque id, recollé
    # print(rebuilt == s, repr(s))
