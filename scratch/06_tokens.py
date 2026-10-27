from llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()
ids = model.encode("What is the sum of 265 and 345?")[0].tolist()
path_vocab = model.get_path_to_vocab_file()
with open(path_vocab, "r", encoding="utf-8") as ft:
    myDict = json.load(ft)

id_to_text = {token_id: text for text, token_id in myDict.items()}

for id in ids:
    print(f"ID:{id}, Token: {id_to_text[id]}")

