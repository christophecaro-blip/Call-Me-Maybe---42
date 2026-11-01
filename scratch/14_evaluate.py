import json
import re
import sys
from pathlib import Path

# Usage: python -m scratch.14_evaluate [results.json] [expected.json]
RESULTS = Path(sys.argv[1] if len(sys.argv) > 1
               else "data/output/function_calling_results.json")
EXPECTED = Path(sys.argv[2] if len(sys.argv) > 2 else "tests/expected.json")
REGEX_FN = "fn_substitute_string_with_regex"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_regex(params):
    """Return the substituted text, or None if the regex is invalid."""
    try:
        return re.sub(params["regex"], params["replacement"], params["source_string"])
    except re.error:
        return None


def params_ok(name, got, want):
    if name == REGEX_FN:
        return apply_regex(got) == apply_regex(want)
    return got == want


results = {r["prompt"]: r for r in load(RESULTS)}
expected = load(EXPECTED)
names_ok = 0
calls_ok = 0
errors = []

for want in expected:
    got = results.get(want["prompt"])
    if got is None:
        errors.append(f"MISSING  {want['prompt']}")
        continue
    if got["name"] != want["name"]:
        errors.append(f"NAME  {want['prom']}: {got ['name']}, want {want['name']}")
        continue
    names_ok += 1
    if params_ok(got["name"], got["parameters"], want["parameters"]):
        calls_ok += 1
    else:
        errors.append(f"PARAMS   {want['prompt']}\n  got:  {got['parameters']}\n  want: {want['parameters']}")

total = len(expected)
print(f"names: {names_ok}/{total}  calls: {calls_ok}/{total} ({100 * calls_ok / total:.0f}%)")
for e in errors:
    print(e)
