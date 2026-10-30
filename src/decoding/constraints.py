"""Rules that decide which text may come next in a generated value.

A constraint is a small state machine. The decoder asks it whether a
token's text is allowed, records the chosen text, and stops once the
value is complete. The model picks among allowed tokens; the constraint
only forbids.
"""

from enum import Enum, auto

from pydantic import BaseModel

from src.errors import DecodingError
from src.models import ParameterSpec


class Constraint(BaseModel):
    """Base class: the contract shared by every value type."""

    def accepts(self, text: str) -> bool:
        """Tell whether `text` can extend the value, without changing it.

        Args:
            text: text of a candidate token.

        Returns:
            True if the value stays valid with `text` appended.
        """
        raise NotImplementedError

    def advance(self, text: str) -> None:
        """Record the text of the chosen token.

        Args:
            text: text of the token the decoder kept; must be accepted.
        """
        raise NotImplementedError

    def is_complete(self) -> bool:
        """Tell whether the value is finished and generation can stop."""
        raise NotImplementedError


class ChoiceConstraint(Constraint):
    """Pick one of a fixed list of strings, then an end marker.

    Used for function names (end `"`, the opening quote being forced
    before) and booleans (end `,` or `}`, the next structural char).
    Nothing may follow the end marker.

    Attributes:
        options: allowed values (e.g. the function names).
        generated: text accepted so far.
        end: text that closes the value; chosen by the caller, which
            knows what comes next in the JSON.
    """

    options: list[str]
    generated: str = ""
    end: str = '"'

    def accepts(self, text: str) -> bool:
        """Accept a start of an option, or a full option plus `end`.

        When an option is a prefix of another ("fn_add" and
        "fn_add_numbers"), both continuing and closing are allowed:
        the model's logits decide.
        """
        candidate = self.generated + text
        is_prefix = any(
            option.startswith(candidate) for option in self.options)
        is_closed = any(
            option + self.end == candidate for option in self.options)

        return is_prefix or is_closed

    def advance(self, text: str) -> None:
        """Append the chosen text to what was generated."""
        self.generated += text

    def is_complete(self) -> bool:
        """Done once a full option is followed by the end marker."""
        return any(
            option + self.end == self.generated for option in self.options)


class NumberState(Enum):
    """Where the number automaton is, after the characters read so far."""

    START = auto()
    SIGN = auto()
    ZERO = auto()
    INT = auto()
    DOT = auto()
    FRAC = auto()


# States where the number may stop: "0", "26", "3.14" (not "-" or "1.").
ACCEPTING = {NumberState.ZERO, NumberState.INT, NumberState.FRAC}


def _step(state: NumberState, char: str) -> NumberState | None:
    """Read one character of a number (end marker excluded).

    Args:
        state: current state.
        char: next character.

    Returns:
        The new state, or None if `char` cannot come next.
    """
    if state is NumberState.START:
        if char == "-":
            return NumberState.SIGN
        if char == "0":
            return NumberState.ZERO
        if char in "123456789":
            return NumberState.INT
        return None
    if state is NumberState.INT:
        if char == ".":
            return NumberState.DOT
        if char in "1234567890":
            return NumberState.INT
        return None
    if state is NumberState.DOT:
        if char in "1234567890":
            return NumberState.FRAC
        return None
    if state is NumberState.SIGN:
        if char in "123456789":
            return NumberState.INT
        if char == "0":
            return NumberState.ZERO
        return None
    if state is NumberState.ZERO:
        if char == ".":
            return NumberState.DOT
        return None
    if state is NumberState.FRAC:
        if char == "0":
            return NumberState.FRAC
        if char in "123456789":
            return NumberState.FRAC
        return None
    return None


class NumberConstraint(Constraint):
    """A JSON number (no exponent), closed by the end marker.

    Grammar: -?(0|[1-9][0-9]*)(\\.[0-9]+)? then `end`. The marker is only
    allowed in an accepting state, so "-" or "1." can never be closed.
    Past `max_length` characters, an accepting state only allows the
    marker; a non-accepting one ("-", "12.") allows one more digit. So
    the number always ends and never gets stuck.

    Attributes:
        end: "," or "}", chosen by the caller (what follows in the JSON).
        max_length: maximum number of characters of the number itself.
        state: automaton state after the characters read so far.
        length: number of characters of the number read so far.
        done: True once the end marker has been read.
        allow_fraction: False for "integer" (the "." is refused).
    """

    end: str
    max_length: int = 20
    state: NumberState = NumberState.START
    length: int = 0
    done: bool = False
    allow_fraction: bool = True

    def _run(self, text: str) -> tuple[NumberState, int, bool] | None:
        """Simulate `text` from the current state, without changing it.

        Args:
            text: text of a candidate token.

        Returns:
            The (state, length, done) reached, or None if refused.
        """
        state, length, done = self.state, self.length, self.done
        for char in text:
            if done:
                return None
            if char == self.end:
                if state not in ACCEPTING:
                    return None
                done = True
            elif length >= self.max_length and state in ACCEPTING:
                return None
            elif char == "." and not self.allow_fraction:
                return None
            else:
                next_state = _step(state, char)
                if next_state is None:
                    return None
                state = next_state
                length += 1
        return state, length, done

    def accepts(self, text: str) -> bool:
        """Accept number characters, then the marker once complete."""
        return self._run(text) is not None

    def advance(self, text: str) -> None:
        """Record the chosen text by updating the automaton.

        Raises:
            DecodingError: `text` is refused, i.e. the decoder did not
                check `accepts` first (a bug).
        """
        result = self._run(text)
        if result is None:
            raise DecodingError(f"number constraint refused {text!r}")
        self.state, self.length, self.done = result

    def is_complete(self) -> bool:
        """Done once the end marker has been read."""
        return self.done


# Letters allowed after a backslash in JSON (\uXXXX is not supported).
JSON_ESCAPES = '"\\/bfnrt'


class StringConstraint(Constraint):
    """The content of a JSON string, then its closing quote.

    The opening quote is forced before. Any character is allowed except
    raw control characters (code < 32, e.g. a newline). A backslash
    opens an escape: only a letter of JSON_ESCAPES may follow, so the
    model must write \\\\d for a regex \\d. The quote may come at once:
    "" is valid. Past `max_length` characters only the quote is allowed,
    but a started escape can always be finished.

    Attributes:
        max_length: maximum number of characters of the content; an
            escape such as \\n counts as one.
        length: number of content characters read so far.
        done: True once the closing quote has been read.
        escaping: True right after a backslash.
    """

    max_length: int = 100
    length: int = 0
    done: bool = False
    escaping: bool = False

    def _run(self, text: str) -> tuple[int, bool, bool] | None:
        """Simulate `text` from the current state, without changing it.

        Args:
            text: text of a candidate token.

        Returns:
            The (length, done, escaping) reached, or None if refused.
        """
        length, done, escaping = self.length, self.done, self.escaping
        for char in text:
            if done:
                return None
            if escaping:
                if char not in JSON_ESCAPES:
                    return None
                escaping = False
                length += 1
            elif char == '"':
                done = True
            elif char == "\\":
                if length >= self.max_length:
                    return None
                escaping = True
            elif ord(char) < 32:
                return None
            elif length >= self.max_length:
                return None
            else:
                length += 1
        return length, done, escaping

    def accepts(self, text: str) -> bool:
        """Accept content characters, or the closing quote."""
        return self._run(text) is not None

    def advance(self, text: str) -> None:
        """Record the chosen text.

        Raises:
            DecodingError: `text` is refused, i.e. the decoder did not
                check `accepts` first (a bug).
        """
        result = self._run(text)
        if result is None:
            raise DecodingError(f"string constraint refused {text!r}")
        self.length, self.done, self.escaping = result

    def is_complete(self) -> bool:
        """Done once the closing quote has been read."""
        return self.done


def make_constraint(spec: ParameterSpec, end: str) -> Constraint:
    """Build the constraint for one parameter value.

    Args:
        spec: declared type of the parameter.
        end: what follows the value in the JSON ("," or "}"); used by
            numbers and booleans. A string always ends on its own quote.

    Returns:
        A fresh constraint for that type.

    Raises:
        DecodingError: unknown type. Cannot happen with valid input:
            ParamType already rejects other types at load time.
    """
    if spec.type == "number":
        return NumberConstraint(end=end)
    if spec.type == "integer":
        return NumberConstraint(end=end, allow_fraction=False)
    if spec.type == "boolean":
        return ChoiceConstraint(options=["true", "false"], end=end)
    if spec.type == "string":
        return StringConstraint()
    raise DecodingError(f"unsupported parameter type: {spec.type!r}")
