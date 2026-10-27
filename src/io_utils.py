"""Read the input files and write the output file."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.errors import InputFileError
from src.models import FunctionDefinition, PromptItem, FunctionCall


def _read_json_list(path: Path, what: str) -> list[Any]:
    """Read a JSON file that must contain a non-empty list.

    Args:
        path: file to read.
        what: plural name of the items ("functions", "prompts"),
            used in error messages.

    Raises:
        InputFileError: if the file is unreadable, not valid JSON,
            not a list, or an empty list.
    """
    try:
        with open(path, "r", encoding="utf-8") as ft:
            data = json.load(ft)
    except (OSError, json.JSONDecodeError) as e:
        raise InputFileError(f"{path}: {e}") from e

    if not isinstance(data, list):
        raise InputFileError(
            f"{path}: expected a list of {what}, "
            f"got {type(data).__name__}"
        )
    if not data:
        raise InputFileError(f"{path}: no {what} defined (empty list)")
    return data


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions from a JSON file.

    Raises:
        InputFileError: if the file is unreadable, not valid JSON,
            or does not match the expected schema.
    """
    data = _read_json_list(path, "functions")

    try:
        functions = [FunctionDefinition.model_validate(elem) for elem in data]
    except ValidationError as e:
        raise InputFileError(f"{path}: {e}") from e

    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate the prompts from a JSON file.

    Raises:
        InputFileError: if the file is unreadable, not valid JSON,
            or does not match the expected schema.
    """
    data = _read_json_list(path, "prompts")

    try:
        prompts = [PromptItem.model_validate(elem) for elem in data]
    except ValidationError as e:
        raise InputFileError(f"{path}: {e}") from e

    return prompts


def write_results(path: Path, calls: list[FunctionCall]) -> None:
    """Write the function calls to a JSON file, creating parent folders.

    Raises:
        InputFileError: if the folder or the file cannot be written.
    """
    data = [call.model_dump() for call in calls]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as ft:
            json.dump(data, ft, indent=2, ensure_ascii=False)
    except OSError as e:
        raise InputFileError(f"{path}: {e}") from e
