"""Optional tokenizer wrapper (HuggingFace `tokenizers` library)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...
    @property
    def eos_token_id(self) -> int | None: ...
    @property
    def vocab_size(self) -> int: ...


class HFTokenizer:
    """Thin wrapper around `tokenizers.Tokenizer`."""

    def __init__(self, path: str | Path, eos_token: str = "<|endoftext|>"):
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "tokenizers package required. Install with: "
                "pip install 'spider-web-moe[tokenizers]'"
            ) from e
        self._tok = Tokenizer.from_file(str(path))
        self._eos_token = eos_token
        eid = self._tok.token_to_id(eos_token)
        self._eos_id = eid

    def encode(self, text: str) -> list[int]:
        ids = self._tok.encode(text).ids
        if not ids:
            raise ValueError("prompt encoded to an empty token sequence")
        return list(ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_id

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()


def load_tokenizer(path: str | Path | None) -> HFTokenizer | None:
    if path is None:
        return None
    return HFTokenizer(path)
