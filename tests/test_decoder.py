"""Tests for the constrained decoder, with a fake engine (no model)."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.decoding.constraints import ChoiceConstraint, StringConstraint
from src.decoding.decoder import decode_value
from src.errors import DecodingError
from src.llm.engine import Engine
from src.llm.vocabulary import Vocabulary

TOKENS = ['"', '",', "a", "b", "x"]


class FakeEngine(Engine):
    """An engine whose scores are fixed in advance, step by step."""

    def __init__(self, vocab: Vocabulary, steps: list[list[float]]) -> None:
        """Use `steps[k]` as the logits of the k-th call."""
        self.vocab = vocab
        self._steps = steps
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        """Encode a text that is exactly one fake token."""
        return [TOKENS.index(text)]

    def next_logits(self, ids: list[int]) -> np.ndarray:
        """Return the scores planned for this step."""
        scores = self._steps[min(self.calls, len(self._steps) - 1)]
        self.calls += 1
        return np.array(scores, dtype=np.float32)


@pytest.fixture
def vocab(tmp_path: Path) -> Vocabulary:
    """A five-token vocabulary (all plain ASCII, so no byte mapping)."""
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({t: i for i, t in enumerate(TOKENS)}),
                    encoding="utf-8")
    return Vocabulary(path, len(TOKENS))


def test_picks_best_allowed_token(vocab: Vocabulary) -> None:
    """The refused best token 'x' is skipped for the next one."""
    engine = FakeEngine(vocab, [[0, 0, 5, 1, 9], [0, 0, 1, 5, 9],
                                [9, 0, 1, 1, 1]])
    ids, text = decode_value(engine, [], ChoiceConstraint(options=["ab"]), 5)
    assert text == 'ab"'
    assert ids == [2, 3, 0]


def test_overflowing_token_is_cut_to_its_closing_start(
    vocab: Vocabulary,
) -> None:
    """'",' is refused, but its start '"' closes the string: kept."""
    engine = FakeEngine(vocab, [[0, 0, 5, 1, 1], [1, 9, 5, 1, 1]])
    constraint = StringConstraint()
    ids, text = decode_value(engine, [], constraint, 5)
    assert text == 'a"'
    assert ids == [2, 0]
    assert constraint.is_complete()


def test_cut_only_when_the_start_ends_the_value(vocab: Vocabulary) -> None:
    """A refused token is never cut to a mere partial value."""
    engine = FakeEngine(vocab, [[0, 0, 1, 1, 9], [9, 0, 1, 1, 1]])
    ids, text = decode_value(engine, [], ChoiceConstraint(options=["a"]), 5)
    assert text == 'a"'


def test_max_tokens_stops_the_loop(vocab: Vocabulary) -> None:
    """The loop ends after max_tokens, even if the value is not done."""
    engine = FakeEngine(vocab, [[0, 0, 9, 1, 1]])
    constraint = StringConstraint()
    ids, text = decode_value(engine, [], constraint, 3)
    assert text == "aaa"
    assert not constraint.is_complete()


def test_input_ids_are_not_modified(vocab: Vocabulary) -> None:
    """The caller's list is copied, not extended in place."""
    engine = FakeEngine(vocab, [[9, 0, 0, 0, 0]])
    prompt_ids = [4, 4]
    ids, _ = decode_value(engine, prompt_ids, StringConstraint(), 5)
    assert prompt_ids == [4, 4]
    assert ids == [4, 4, 0]


def test_stuck_constraint_raises(vocab: Vocabulary) -> None:
    """No allowed token at all is a bug, reported clearly."""
    engine = FakeEngine(vocab, [[1, 1, 1, 1, 1]])
    with pytest.raises(DecodingError):
        decode_value(engine, [], ChoiceConstraint(options=["zzz"]), 5)
