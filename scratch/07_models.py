import json
import copy

from pydantic import ValidationError

from src.models import FunctionCall, FunctionDefinition, PromptItem

FUNCTIONS_PATH = "data/input/functions_definition.json"
PROMPTS_PATH = "data/input/function_calling_tests.json"


def main() -> None:
    with open(FUNCTIONS_PATH, encoding="utf-8") as f:
        raw_functions = json.load(f)
    functions = [FunctionDefinition.model_validate(elem) for elem in raw_functions]
    for fn in functions:
        print(fn.name, list(fn.parameters), fn.returns.type)

    # 2. Même chose pour les prompts → afficher combien il y en a
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        raw_prompts = json.load(f)
    prompts = [PromptItem.model_validate(elem) for elem in raw_prompts]
    print(len(prompts))

    # 3. Fabriquer un FunctionCall à la main, puis l'afficher avec model_dump()
    call = FunctionCall(prompt="What is the sum of 2 and 3?", name="fn_add_numbers", parameters={"a": 2, "b": 3.0})
    print(call.model_dump())

    # # 4. Casser une définition exprès et observer l'erreur
    broken = copy.deepcopy(raw_functions[0])
    broken["parameters"]["a"]["type"] = "banana"
    try:
        FunctionDefinition.model_validate(broken)
    except ValidationError as e:
        print(e)
    
    # 5. 
    broken = copy.deepcopy(raw_functions[0])
    del broken["description"]
    try:
        FunctionDefinition.model_validate(broken)
    except ValidationError as e:
        print(e)


if __name__ == "__main__":
    main()
