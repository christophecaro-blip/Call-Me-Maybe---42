"""Map every token id to the exact text it adds to the output."""

import json
from pathlib import Path


def _bytes_to_unicode() -> dict[int, str]:
    """Build GPT-2's table that maps each of the 256 bytes to a printable char.

    Printable bytes keep their own character ("a" stays "a"). The others
    (space, newline, control bytes...) get chr(256), chr(257)... in order,
    so space (32) becomes "Ġ" (chr(288)) and newline (10) becomes "Ċ".
    """
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    code_points = byte_values[:]
    shifted = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            code_points.append(256 + shifted)
            shifted += 1
    return dict(zip(byte_values, map(chr, code_points)))


def _decode_token(token: str, char_to_byte: dict[str, int]) -> str | None:
    """Return the real text of a vocab.json token.

    Args:
        token: token as written in vocab.json (e.g. "Ġthe").
        char_to_byte: inverse of _bytes_to_unicode().

    Returns:
        The decoded text (e.g. " the"), or None if the token is only a
        fragment of a multi-byte UTF-8 character.
    """
    raw = bytes(char_to_byte[c] for c in token)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


class Vocabulary:
    """Text added to the output by each token id.

    Attributes:
        id_to_text: one entry per logit; the token's text, or None if the
            id must never be generated (special token, UTF-8 fragment,
            or id without a token).
    """

    def __init__(self, vocab_path: Path, size: int) -> None:
        """Load vocab.json and decode every token.

        Args:
            vocab_path: path to the tokenizer's vocab.json.
            size: number of logits returned by the model (151 936 for
                Qwen3-0.6B), which is larger than the vocabulary.
        """
        char_to_byte = {v: k for k, v in _bytes_to_unicode().items()}
        self.id_to_text: list[str | None] = [None] * size
        with open(vocab_path, encoding="utf-8") as f:
            vocab_json = json.load(f)
        for token, token_id in vocab_json.items():
            self.id_to_text[token_id] = _decode_token(token, char_to_byte)
