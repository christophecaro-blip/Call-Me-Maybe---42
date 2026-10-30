"""Program entry point: read the inputs, build the calls, write the output."""

import sys

from src.cli import parse_args
from src.errors import CallMeMaybeError, DecodingError
from src.io_utils import load_functions, load_prompts, write_results
from src.llm.engine import Engine
from src.models import FunctionCall, FunctionDefinition, PromptItem
from src.pipeline import FunctionCaller


def _run_pipeline(
    functions: list[FunctionDefinition], prompts: list[PromptItem]
) -> list[FunctionCall]:
    """Generate one call per prompt with the model.

    The model is loaded once. A prompt that fails is reported on stderr
    and left out of the output, rather than replaced by a made-up call;
    the other prompts are still processed.
    """
    caller = FunctionCaller(engine=Engine(), functions=functions)
    calls = []
    for item in prompts:
        try:
            calls.append(caller.call(item.prompt))
        except DecodingError as e:
            print(f"Warning: skipped {item.prompt!r}: {e}", file=sys.stderr)
    return calls


def main() -> int:
    """Run the whole program and return the process exit code.

    Returns:
        0 on success, 1 on a reported error, 130 if interrupted (Ctrl+C).
    """
    try:
        config = parse_args()
        functions = load_functions(config.functions_definition)
        prompts = load_prompts(config.input)
        calls = _run_pipeline(functions, prompts)
        write_results(config.output, calls)
        return 0
    except CallMeMaybeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Error: Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
