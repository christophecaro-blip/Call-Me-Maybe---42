"""Tests for the decoding constraints (no model needed)."""

import json

import pytest

from src.decoding.constraints import (
    ChoiceConstraint,
    Constraint,
    NumberConstraint,
    StringConstraint,
    make_constraint,
)
from src.models import ParameterSpec

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


@pytest.fixture
def boolean() -> ChoiceConstraint:
    """A boolean value followed by another parameter (ends on ',')."""
    return ChoiceConstraint(options=["true", "false"], end=",")


def test_custom_end_marker(boolean: ChoiceConstraint) -> None:
    """The end marker replaces the quote: 'true,' ends, 'true"' does not."""
    assert boolean.accepts("true,")
    assert not boolean.accepts('true"')
    boolean.advance("true,")
    assert boolean.is_complete()


def test_boolean_split_tokens(boolean: ChoiceConstraint) -> None:
    """The value and its marker may come in separate tokens."""
    boolean.advance("true")
    assert not boolean.is_complete()
    assert boolean.accepts(",")
    boolean.advance(",")
    assert boolean.is_complete()


def test_boolean_overflow_refused(boolean: ChoiceConstraint) -> None:
    """Nothing may follow the end marker, not even a space."""
    assert not boolean.accepts("true, ")
    assert not boolean.accepts("false},")


def test_default_end_is_quote() -> None:
    """Without `end`, the closing quote is used (phase 4 unchanged)."""
    assert ChoiceConstraint(options=["fn"]).end == '"'


# --- NumberConstraint -------------------------------------------------


def _feed(constraint: Constraint, *tokens: str) -> None:
    """Advance through tokens, checking each one is accepted first."""
    for token in tokens:
        assert constraint.accepts(token), token
        constraint.advance(token)


@pytest.mark.parametrize("text", ["0", "-3", "3.14", "-0.5", "265", "10"])
def test_number_valid_then_end(text: str) -> None:
    """Valid JSON numbers can be closed by the end marker."""
    number = NumberConstraint(end=",")
    _feed(number, text)
    assert not number.is_complete()
    _feed(number, ",")
    assert number.is_complete()


def test_number_split_tokens() -> None:
    """A number may arrive in many tokens, ending in "5," style."""
    number = NumberConstraint(end="}")
    _feed(number, "3", ".", "1", "4}")
    assert number.is_complete()


@pytest.mark.parametrize("text", ["007", "01", "-a", "1..2", "--1", "+1"])
def test_number_invalid_refused(text: str) -> None:
    """Leading zeros, double signs or dots and letters are refused."""
    assert not NumberConstraint(end=",").accepts(text)


@pytest.mark.parametrize("text", ["", "-", "1."])
def test_number_incomplete_cannot_end(text: str) -> None:
    """The end marker is refused outside an accepting state."""
    number = NumberConstraint(end=",")
    if text:
        _feed(number, text)
    assert not number.accepts(",")


@pytest.mark.parametrize("text", [" 5", "5 ", "5, ", "5,6"])
def test_number_spaces_and_overflow_refused(text: str) -> None:
    """No spaces, and nothing after the end marker."""
    assert not NumberConstraint(end=",").accepts(text)


def test_number_wrong_end_marker_refused() -> None:
    """Only the configured end marker closes the number."""
    assert not NumberConstraint(end=",").accepts("5}")


def test_number_accepts_does_not_change_state() -> None:
    """Asking is free: a refused '.' after '1.' changes nothing."""
    number = NumberConstraint(end=",")
    _feed(number, "1.")
    number.accepts("5")
    assert not number.accepts(",")


def test_number_max_length_forces_end() -> None:
    """Past max_length, only the end marker is allowed."""
    number = NumberConstraint(end=",", max_length=3)
    _feed(number, "123")
    assert not number.accepts("4")
    assert number.accepts(",")


@pytest.mark.parametrize("text", ["12.", "-"])
def test_number_max_length_never_stuck(text: str) -> None:
    """At the limit in a non-accepting state, one more digit is allowed."""
    number = NumberConstraint(end=",", max_length=len(text))
    _feed(number, text)
    assert number.accepts("5")
    _feed(number, "5")
    assert not number.accepts("6")
    assert number.accepts(",")


@pytest.mark.parametrize("text", ["0", "-7", "42"])
def test_integer_valid(text: str) -> None:
    """Without fractions, whole numbers still work."""
    integer = NumberConstraint(end="}", allow_fraction=False)
    _feed(integer, text, "}")
    assert integer.is_complete()


@pytest.mark.parametrize("text", ["3.", "0.5", "-1.0"])
def test_integer_refuses_dot(text: str) -> None:
    """With allow_fraction=False, the '.' is refused."""
    assert not NumberConstraint(end=",", allow_fraction=False).accepts(text)


def test_number_allows_fraction_by_default() -> None:
    """Without the flag, 'number' keeps its decimal part."""
    assert NumberConstraint(end=",").allow_fraction


# --- StringConstraint -------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["shrek", "hello world", "it's 'quoted'", "Zoé", "3 + 4 = 7", "a-b_c!?"],
)
def test_string_content_then_quote(text: str) -> None:
    """Ordinary text (spaces, apostrophes, accents) is accepted."""
    string = StringConstraint()
    _feed(string, text)
    assert not string.is_complete()
    _feed(string, '"')
    assert string.is_complete()


def test_string_empty_is_valid() -> None:
    """The closing quote is allowed right away: "" is valid JSON."""
    string = StringConstraint()
    _feed(string, '"')
    assert string.is_complete()


def test_string_split_tokens() -> None:
    """Content and closing quote may come in any split."""
    string = StringConstraint()
    _feed(string, " The", " cat", ' sat"')
    assert string.is_complete()


@pytest.mark.parametrize("text", ["line\nbreak", "tab\there", "\x00"])
def test_string_controls_refused(text: str) -> None:
    """Raw control characters (code < 32) are refused."""
    assert not StringConstraint().accepts(text)


@pytest.mark.parametrize("text", ['abc",', 'abc" ', 'abc"}', '""'])
def test_string_overflow_refused(text: str) -> None:
    """Nothing may follow the closing quote."""
    assert not StringConstraint().accepts(text)


def test_string_accepts_does_not_change_state() -> None:
    """Asking is free: only advance() changes the state."""
    string = StringConstraint()
    string.accepts('abc"')
    assert not string.is_complete()
    assert string.length == 0


def test_string_max_length_forces_quote() -> None:
    """Past max_length, only the closing quote is allowed."""
    string = StringConstraint(max_length=3)
    _feed(string, "abc")
    assert not string.accepts("d")
    assert string.accepts('"')


def test_string_default_max_length() -> None:
    """The default limit is large enough for the test prompts."""
    assert StringConstraint().max_length == 100


@pytest.mark.parametrize("escape", ["\\\\", '\\"', "\\/", "\\b", "\\f",
                                    "\\n", "\\r", "\\t"])
def test_string_valid_escapes(escape: str) -> None:
    """The eight JSON escapes are accepted and do not end the string."""
    string = StringConstraint()
    _feed(string, "a" + escape)
    assert not string.is_complete()
    _feed(string, '"')
    assert string.is_complete()


@pytest.mark.parametrize("text", ["\\d", "\\x", "\\u00e9", "\\ ", "\\a"])
def test_string_invalid_escapes_refused(text: str) -> None:
    """Letters that JSON does not define (and \\u, not supported)."""
    assert not StringConstraint().accepts(text)


def test_string_escaped_quote_does_not_close() -> None:
    """In \\" the quote is part of the text, not the end."""
    string = StringConstraint()
    _feed(string, 'say \\"hi\\"')
    assert not string.is_complete()
    _feed(string, '"')
    assert string.is_complete()


def test_string_regex_round_trip() -> None:
    """The model must write \\\\d+ so that json.loads gives \\d+."""
    string = StringConstraint()
    _feed(string, "\\", "\\", "d", "+", '"')
    assert string.is_complete()
    assert json.loads('"' + "\\\\d+" + '"') == "\\d+"
    assert not StringConstraint().accepts("\\d")


def test_string_escape_letter_never_blocked_by_limit() -> None:
    """At the limit, a pending escape can still be finished."""
    string = StringConstraint(max_length=2)
    _feed(string, "a", "\\")
    _feed(string, "n")
    assert not string.accepts("b")
    assert string.accepts('"')


def test_string_backslash_refused_at_limit() -> None:
    """At the limit, a new escape is refused; the quote still works."""
    string = StringConstraint(max_length=1)
    _feed(string, "a")
    assert not string.accepts("\\")
    assert string.accepts('"')


# --- make_constraint --------------------------------------------------

def test_factory_number() -> None:
    """'number' gives a number constraint that allows decimals."""
    constraint = make_constraint(ParameterSpec(type="number"), ",")
    assert isinstance(constraint, NumberConstraint)
    _feed(constraint, "2.5,")
    assert constraint.is_complete()


def test_factory_integer() -> None:
    """'integer' gives a number constraint without decimals."""
    constraint = make_constraint(ParameterSpec(type="integer"), ",")
    assert isinstance(constraint, NumberConstraint)
    assert constraint.accepts("42,")
    assert not constraint.accepts("2.5")


def test_factory_boolean() -> None:
    """'boolean' gives a choice between true and false."""
    constraint = make_constraint(ParameterSpec(type="boolean"), "}")
    assert isinstance(constraint, ChoiceConstraint)
    assert constraint.accepts("false}")
    assert not constraint.accepts("yes")


def test_factory_string_ignores_end() -> None:
    """'string' ends on its own quote: a ',' is just content."""
    constraint = make_constraint(ParameterSpec(type="string"), ",")
    assert isinstance(constraint, StringConstraint)
    _feed(constraint, "shrek,")
    assert not constraint.is_complete()
    _feed(constraint, '"')
    assert constraint.is_complete()


@pytest.mark.parametrize("end", [",", "}"])
def test_factory_passes_end(end: str) -> None:
    """The end marker given by the caller is the one used."""
    constraint = make_constraint(ParameterSpec(type="number"), end)
    assert constraint.accepts("7" + end)
