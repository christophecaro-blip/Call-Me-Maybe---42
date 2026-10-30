from pathlib import Path

from src.io_utils import load_functions, load_prompts
from src.llm.engine import Engine
from src.pipeline import FunctionCaller

functions = load_functions(Path("data/input/functions_definition.json"))
prompts = load_prompts(Path("data/input/function_calling_tests.json"))
caller = FunctionCaller(engine=Engine(), functions=functions)

for item in prompts:
    print(caller.call(item.prompt).model_dump())
