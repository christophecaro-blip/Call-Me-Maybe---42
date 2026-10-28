"""Single entry point to the language model.

Only this module imports llm_sdk. The rest of the program talks to Engine
with plain types (list[int], np.ndarray), never to the SDK directly.
"""

from pathlib import Path

import numpy as np
from llm_sdk import Small_LLM_Model

from src.llm.vocabulary import Vocabulary


class Engine:
    """Thin wrapper around the SDK model.

    Attributes:
        vocab: text of every token id, sized to the model's logits.
    """

    def __init__(self) -> None:
        """Load the model and build its vocabulary.

        The logits size is measured once with a first call, so it is
        never hardcoded.
        """
        self._model = Small_LLM_Model()
        size = len(self.next_logits(self.encode("Hello")))
        vocab_path = Path(self._model.get_path_to_vocab_file())
        self.vocab = Vocabulary(vocab_path, size)

    def encode(self, text: str) -> list[int]:
        """Split text into token ids.

        Args:
            text: text to tokenize.

        Returns:
            The token ids, as a flat Python list.
        """
        return self._model.encode(text)[0].tolist()

    def next_logits(self, ids: list[int]) -> np.ndarray:
        """Score every possible next token.

        Args:
            ids: token ids of the text generated so far.

        Returns:
            One float32 score per token id; a higher score means the
            model finds that token more likely.
        """
        logits = self._model.get_logits_from_input_ids(ids)
        return np.array(logits, dtype=np.float32)
