import json

from llm_sdk import Small_LLM_Model


def bytes_to_unicode() -> dict[int, str]:
       bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
            + list(range(ord("®"), ord("ÿ") + 1))
       cs = bs[:]
       n = 0
       for b in range(256):
           if b not in bs:
               bs.append(b)
               cs.append(256 + n)
               n += 1
       return dict(zip(bs, map(chr, cs)))

myDict = bytes_to_unicode()
newDict = {v: k for k, v in myDict.items()}
tok_the = bytes(newDict[c] for c in "Ã©")
# print(len(bytes_to_unicode()))
# print(newDict)

# half = bytes(newDict[c] for c in "Ã")     # seulement le 1er octet de é
# print(half)                                # à quoi ressemble-t-il ?
# # print(half.decode("utf-8"))                # que se passe-t-il ?
# print(half.decode("utf-8", errors="replace"))                # que se passe-t-il ?

# print(tok_the.decode("utf-8"))

model = Small_LLM_Model()
with open(model.get_path_to_vocab_file(), encoding="utf-8") as f:
    vocab = json.load(f)                 # {"Ġthe": 279, ...}

ok = 0
broken = []
for token, token_id in vocab.items():
    raw = bytes(newDict[c] for c in token)
    try:
        raw.decode("utf-8")
        ok += 1
    except UnicodeDecodeError:
        broken.append(token)

print(ok, len(broken))
print(broken[:10])