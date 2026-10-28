"""Tests for the byte-level vocabulary (no model needed)."""

import json
from pathlib import Path

import pytest

from src.llm.vocabulary import Vocabulary, _bytes_to_unicode, _decode_token


@pytest.fixture
def char_to_byte() -> dict[str, int]:
    """Inverse GPT-2 table: printable char -> byte."""
    return {c: b for b, c in _bytes_to_unicode().items()}


def test_table_covers_all_bytes_once() -> None:
    """The 256 bytes map to 256 distinct chars."""
    table = _bytes_to_unicode()
    assert sorted(table) == list(range(256))
    assert len(set(table.values())) == 256


def test_table_known_values() -> None:
    """Printable bytes keep their char; space and newline are shifted."""
    table = _bytes_to_unicode()
    assert table[ord("a")] == "a"
    assert table[ord(" ")] == "Ġ"
    assert table[ord("\n")] == "Ċ"


@pytest.mark.parametrize(
    ("token", "text"),
    [
        ("Ġthe", " the"),
        ("Ċ", "\n"),
        ("7", "7"),
        ("Ã©", "é"),
        ('"', '"'),
    ],
)
def test_decode_token(
    token: str, text: str, char_to_byte: dict[str, int]
) -> None:
    """Complete tokens decode to their real text."""
    assert _decode_token(token, char_to_byte) == text


def test_decode_fragment_is_none(char_to_byte: dict[str, int]) -> None:
    """Half of a multi-byte char is unusable: None, never '�'."""
    assert _decode_token("Ã", char_to_byte) is None


def test_vocabulary_fills_and_leaves_gaps(tmp_path: Path) -> None:
    """Ids missing from vocab.json (special tokens, holes) stay None."""
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(
        json.dumps({"Ġthe": 0, "Ċ": 1, "Ã": 2}), encoding="utf-8"
    )
    vocab = Vocabulary(vocab_file, size=5)

    assert vocab.id_to_text == [" the", "\n", None, None, None]
