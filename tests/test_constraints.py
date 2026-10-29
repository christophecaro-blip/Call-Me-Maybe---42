"""Tests for the decoding constraints (no model needed)."""

import pytest

from src.decoding.constraints import ChoiceConstraint, Constraint

NAMES = ["fn_add", "fn_add_numbers", "fn_greet"]


@pytest.fixture
def choice() -> ChoiceConstraint:
    """A fresh constraint over the test names."""
    return ChoiceConstraint(options=NAMES)


def test_base_class_is_abstract() -> None:
    """The base class only defines the contract."""
    base = Constraint()
    with pytest.raises(NotImplementedError):
        base.accepts("x")


@pytest.mark.parametrize("text", ["f", "fn", "fn_", "fn_gr", "fn_greet"])
def test_accepts_prefixes(choice: ChoiceConstraint, text: str) -> None:
    """Any start of an option is allowed."""
    assert choice.accepts(text)


@pytest.mark.parametrize("text", ["Hello", " fn", "fn_x", '"', "g"])
def test_rejects_non_prefixes(choice: ChoiceConstraint, text: str) -> None:
    """Text that starts no option is refused (even a leading space)."""
    assert not choice.accepts(text)


def test_accepts_does_not_change_state(choice: ChoiceConstraint) -> None:
    """Asking is free: only advance() changes the state."""
    choice.accepts("fn")
    assert choice.generated == ""


def test_shared_prefix_allows_both(choice: ChoiceConstraint) -> None:
    """After "fn_add", continuing and closing are both allowed."""
    choice.advance("fn_add")
    assert choice.accepts("_numbers")
    assert choice.accepts('"')
    assert not choice.is_complete()


def test_closing_quote_completes(choice: ChoiceConstraint) -> None:
    """A full option followed by its quote ends the value."""
    choice.advance("fn_gr")
    choice.advance("eet")
    choice.advance('"')
    assert choice.is_complete()
    assert choice.generated == 'fn_greet"'


def test_overflow_after_quote_is_refused(choice: ChoiceConstraint) -> None:
    """Nothing may follow the closing quote (e.g. a '",' token)."""
    choice.advance("fn_greet")
    assert not choice.accepts('",')
    assert not choice.accepts('" ')


def test_quote_on_partial_name_is_refused(choice: ChoiceConstraint) -> None:
    """Closing in the middle of a name would give an unknown function."""
    choice.advance("fn_gr")
    assert not choice.accepts('"')


def test_full_token_in_one_step(choice: ChoiceConstraint) -> None:
    """A single token may carry the whole name and the quote."""
    assert choice.accepts('fn_greet"')
