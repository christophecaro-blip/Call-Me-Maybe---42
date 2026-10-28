"""Tests for the command-line parsing."""

from pathlib import Path

import pytest

from src.cli import (
    DEFAULT_FUNCTIONS,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    parse_args,
)


def test_defaults() -> None:
    """Without options, the three default paths are used."""
    config = parse_args([])
    assert config.functions_definition == DEFAULT_FUNCTIONS
    assert config.input == DEFAULT_INPUT
    assert config.output == DEFAULT_OUTPUT


def test_override_one_option() -> None:
    """A given option replaces only its own default, as a Path."""
    config = parse_args(["--input", "x.json"])
    assert config.input == Path("x.json")
    assert config.output == DEFAULT_OUTPUT


def test_unknown_option_exits_with_code_2() -> None:
    """argparse rejects unknown options with exit code 2."""
    with pytest.raises(SystemExit) as exc:
        parse_args(["--bad"])
    assert exc.value.code == 2
