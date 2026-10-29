"""Rules that decide which text may come next in a generated value.

A constraint is a small state machine. The decoder asks it whether a
token's text is allowed, records the chosen text, and stops once the
value is complete. The model picks among allowed tokens; the constraint
only forbids.
"""

from pydantic import BaseModel


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
    """Pick one of a fixed list of strings, then its closing quote.

    The opening quote is forced before; this constraint covers the
    option itself and the closing `"`. Nothing may follow the quote.

    Attributes:
        options: allowed values (e.g. the function names).
        generated: text accepted so far.
    """

    options: list[str]
    generated: str = ""

    def accepts(self, text: str) -> bool:
        """Accept a start of an option, or a full option plus `"`.

        When an option is a prefix of another ("fn_add" and
        "fn_add_numbers"), both continuing and closing are allowed:
        the model's logits decide.
        """
        candidate = self.generated + text
        is_prefix = any(
            option.startswith(candidate) for option in self.options)
        is_closed = any(option + '"' == candidate for option in self.options)

        return is_prefix or is_closed

    def advance(self, text: str) -> None:
        """Append the chosen text to what was generated."""
        self.generated += text

    def is_complete(self) -> bool:
        """Done once a full option is followed by its closing quote."""
        return any(option + '"' == self.generated for option in self.options)
