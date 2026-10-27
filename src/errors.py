"""Project exceptions, reported to the user as clean messages."""


class CallMeMaybeError(Exception):
    """Base class for every error the program reports cleanly."""


class InputFileError(CallMeMaybeError):
    """Raised when an input file is missing, invalid JSON or malformed.

    Also raised when the output file cannot be written.
    """


class ModelError(CallMeMaybeError):
    """Raised when the language model cannot be loaded or run."""


class DecodingError(CallMeMaybeError):
    """Raised when constrained decoding cannot produce a valid call."""
