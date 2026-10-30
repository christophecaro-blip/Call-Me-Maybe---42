"""End-to-end tests of the entry point (the model is replaced by a fake)."""

import json
import sys
from pathlib import Path

import pytest

import src.__main__ as entry
from src.models import FunctionCall, FunctionDefinition, PromptItem


def _fake_pipeline(
    functions: list[FunctionDefinition], prompts: list[PromptItem]
) -> list[FunctionCall]:
    """Answer every prompt with the first function, without the model."""
    fn = functions[0]
    return [
        FunctionCall(prompt=p.prompt, name=fn.name,
                     parameters={name: 0.0 for name in fn.parameters})
        for p in prompts
    ]


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    """Call main() as if launched with `args`, with the fake pipeline."""
    monkeypatch.setattr(sys, "argv", ["src", *args])
    monkeypatch.setattr(entry, "_run_pipeline", _fake_pipeline)
    return entry.main()


def test_success_writes_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A normal run returns 0 and writes one call per prompt."""
    out = tmp_path / "results.json"
    assert _run(monkeypatch, "--output", str(out)) == 0

    results = json.loads(out.read_text(encoding="utf-8"))
    assert len(results) == 11
    assert set(results[0]) == {"prompt", "name", "parameters"}


def test_missing_input_returns_1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing input file gives a clean message on stderr and code 1."""
    code = _run(monkeypatch, "--input", str(tmp_path / "nope.json"))
    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("Error:")
    assert "Traceback" not in err
