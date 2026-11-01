"""Constrained greedy decoding: the model scores, the constraint filters.

At each step the best-scored token whose text the constraint accepts is
kept. Invalid output is impossible by construction.
"""

import numpy as np

from src.decoding.constraints import Constraint
from src.errors import DecodingError
from src.llm.engine import Engine


def _closing_prefix(text: str, constraint: Constraint) -> str | None:
    """Return the longest start of `text` that completes the value.

    Used on a refused token such as '",' (closing quote then comma): the
    model wants to close the value, but the comma overflows. Keeping
    only '"' respects that choice without breaking the constraint.

    Returns:
        The start, or None if no start of `text` ends the value.
    """
    for size in range(len(text) - 1, 0, -1):
        prefix = text[:size]
        if not constraint.accepts(prefix):
            continue
        trial = constraint.model_copy(deep=True)
        trial.advance(prefix)
        if trial.is_complete():
            return prefix
    return None


def _best_allowed(logits: np.ndarray, engine: Engine,
                  constraint: Constraint) -> tuple[list[int], str]:
    """Return the best-scored token the constraint accepts.

    Candidates are tried from the highest logit down, so the search
    usually stops within the first few ids (lazy search, see D4). A
    refused token whose start completes the value is cut to that start:
    the model voted to close.

    Args:
        logits: one score per token id.
        engine: gives the text of each id.
        constraint: decides which text may come next.

    Returns:
        The chosen ids and their text.

    Raises:
        DecodingError: no token is allowed, i.e. the constraint is stuck.
    """
    for token_id in np.argsort(logits)[::-1]:
        text = engine.vocab.id_to_text[token_id]
        if text is None:
            continue
        if constraint.accepts(text):
            return [int(token_id)], text
        prefix = _closing_prefix(text, constraint)
        if prefix is not None:
            return engine.encode(prefix), prefix
    raise DecodingError("no token is allowed by the constraint")


def decode_value(engine: Engine, ids: list[int], constraint: Constraint,
                 max_tokens: int) -> tuple[list[int], str]:
    """Generate one value token by token under a constraint.

    Args:
        engine: the model.
        ids: prompt and forced text so far; not modified (copied).
        constraint: rules of the value; updated as tokens are chosen.
        max_tokens: hard limit on generated tokens, so the loop ends.

    Returns:
        The extended ids and the text generated for this value.

    Raises:
        DecodingError: the constraint allows no token at some step.
    """
    ids = list(ids)
    generated = ""
    for _ in range(max_tokens):
        logits = engine.next_logits(ids)
        token_ids, text = _best_allowed(logits, engine, constraint)
        constraint.advance(text)
        generated += text
        ids.extend(token_ids)
        if constraint.is_complete():
            break
    return ids, generated
