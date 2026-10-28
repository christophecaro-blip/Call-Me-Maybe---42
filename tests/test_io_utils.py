"""Tests for the input/output helpers."""

import json
from pathlib import Path

import pytest

from src.errors import InputFileError
from src.io_utils import load_functions, load_prompts, write_results
from src.models import FunctionCall

FUNCTIONS_FILE = Path("data/input/functions_definition.json")
PROMPTS_FILE = Path("data/input/function_calling_tests.json")


def _write(tmp_path: Path, content: str) -> Path:
    """Write `content` to a temporary JSON file and return its path."""
    path = tmp_path / "input.json"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_functions_real_file() -> None:
    """The provided definitions file loads 5 functions."""
    functions = load_functions(FUNCTIONS_FILE)
    assert len(functions) == 5
    assert functions[0].name == "fn_add_numbers"


def test_load_prompts_real_file() -> None:
    """The provided prompts file loads 11 prompts."""
    assert len(load_prompts(PROMPTS_FILE)) == 11


def test_missing_file(tmp_path: Path) -> None:
    """A file that does not exist raises InputFileError."""
    with pytest.raises(InputFileError, match="nope.json"):
        load_functions(tmp_path / "nope.json")


def test_invalid_json(tmp_path: Path) -> None:
    """A malformed JSON file raises InputFileError."""
    with pytest.raises(InputFileError):
        load_functions(_write(tmp_path, "[1,]"))


def test_not_a_list(tmp_path: Path) -> None:
    """A JSON object instead of a list is rejected with a clear message."""
    with pytest.raises(InputFileError, match="expected a list"):
        load_functions(_write(tmp_path, '{"name": "fn"}'))


def test_empty_list(tmp_path: Path) -> None:
    """An empty list is rejected for both input files."""
    path = _write(tmp_path, "[]")
    with pytest.raises(InputFileError, match="no functions"):
        load_functions(path)
    with pytest.raises(InputFileError, match="no prompts"):
        load_prompts(path)


def test_bad_schema(tmp_path: Path) -> None:
    """A function definition with missing fields is rejected."""
    with pytest.raises(InputFileError):
        load_functions(_write(tmp_path, '[{"name": "fn"}]'))


def test_unknown_parameter_type(tmp_path: Path) -> None:
    """A parameter type outside the allowed list is rejected."""
    content = json.dumps([{
        "name": "fn",
        "description": "d",
        "parameters": {"a": {"type": "banana"}},
        "returns": {"type": "number"},
    }])
    with pytest.raises(InputFileError, match="parameters.a.type"):
        load_functions(_write(tmp_path, content))


def test_write_results_creates_folders(tmp_path: Path) -> None:
    """Missing parent folders are created and the content round-trips."""
    out = tmp_path / "a" / "b" / "results.json"
    call = FunctionCall(prompt="Greet Zoé", name="fn_greet",
                        parameters={"name": "Zoé"})
    write_results(out, [call])

    text = out.read_text(encoding="utf-8")
    assert "Zoé" in text                      # ensure_ascii=False
    assert json.loads(text) == [call.model_dump()]


def test_write_results_unwritable(tmp_path: Path) -> None:
    """Writing "inside" an existing file raises InputFileError."""
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(InputFileError):
        write_results(blocker / "results.json", [])
