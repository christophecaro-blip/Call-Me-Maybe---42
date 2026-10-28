"""Parse the command-line options into a typed configuration."""

import argparse
from pathlib import Path

from pydantic import BaseModel

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


class CliConfig(BaseModel):
    """File paths chosen on the command line (or their defaults)."""

    functions_definition: Path
    input: Path
    output: Path


def parse_args(argv: list[str] | None = None) -> CliConfig:
    """Read the command-line options and return them as a CliConfig.

    Args:
        argv: options to parse, without the program name.
            If None, argparse reads the real command line (sys.argv).
    """
    parser = argparse.ArgumentParser(
        description="Turn natural-language prompts into JSON function "
        "calls, using constrained decoding on a small LLM."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="JSON file listing the available functions",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSON file listing the prompts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON file where the function calls are written",
    )
    args = parser.parse_args(argv)

    return CliConfig(
        functions_definition=args.functions_definition,
        input=args.input,
        output=args.output,
    )
