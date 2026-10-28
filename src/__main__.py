"""Program entry point: read the inputs, build the calls, write the output."""

import sys

from src.cli import parse_args
from src.errors import CallMeMaybeError
from src.io_utils import load_functions, load_prompts, write_results
from src.models import FunctionCall, FunctionDefinition, PromptItem


def _fake_calls(
    functions: list[FunctionDefinition], prompts: list[PromptItem]
) -> list[FunctionCall]:
    """Placeholder pipeline: answer every prompt with the first function.

    Every parameter gets a dummy value. Replaced by the real pipeline
    in phase 6.
    """
    fn = functions[0]
    return [
        FunctionCall(
            prompt=p.prompt,
            name=fn.name,
            parameters={name: 0.0 for name in fn.parameters},
        )
        for p in prompts
    ]


def main() -> int:
    """Run the whole program and return the process exit code.

    Returns:
        0 on success, 1 on a reported error, 130 if interrupted (Ctrl+C).
    """
    try:
        config = parse_args()
        functions = load_functions(config.functions_definition)
        prompts = load_prompts(config.input)
        calls = _fake_calls(functions, prompts)
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
