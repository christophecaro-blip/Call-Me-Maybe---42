import json

prefix = '{"name": "fn_greet", "parameters": {"name": "shrek'
tokens = ['"', ':"', '",', '"}', '\\"', ' "']  # Ġ" → ' "' ; \" → '\\"'
endings = ['"}}', '}}', '}', '', ' "x": 1}}']

for tok in tokens:
    text = prefix + tok
    valid_endings = []
    for ending in endings:
        try:
            json.loads(text + ending)
            valid_endings.append(ending)
        except json.JSONDecodeError:
            pass
    print(repr(tok), "->", valid_endings or "aucune fin valide")