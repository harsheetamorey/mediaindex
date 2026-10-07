"""Immutable index profiles.

A profile pins everything that determines which vector space an embedding lives in.
Vectors from different profiles must never be ranked together.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"

QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "


class ProfileMismatch(Exception):
    """Raised when vectors from incompatible profiles would be mixed."""


@dataclass(frozen=True)
class IndexProfile:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    dim: int = 768
    precision: str = "bfloat16"  # bfloat16 | float32; float16 is prohibited by the model card
    encoders: tuple[str, ...] = ("text", "image")
    image_max_soft_tokens: int = 280
    audio_sample_rate: int = 16000
    query_prompt: str = QUERY_PROMPT
    document_prompt: str = DOCUMENT_PROMPT
    media_prompt: str = ""  # media inputs take no prefix (model card)
    normalization: str = "l2-float32"
    extra: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self) -> None:
        if self.precision not in ("bfloat16", "float32"):
            raise ValueError(f"unsupported precision {self.precision!r}; float16 is prohibited")
        if self.dim not in (768, 512, 256, 128):
            raise ValueError(f"unsupported MRL dimension {self.dim}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["encoders"] = sorted(self.encoders)
        d["extra"] = sorted([list(x) for x in self.extra])
        return d

    @property
    def key(self) -> str:
        """Stable content hash; equal keys mean vectors are comparable."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "IndexProfile":
        d = dict(d)
        d["encoders"] = tuple(d.get("encoders", ()))
        d["extra"] = tuple(tuple(x) for x in d.get("extra", ()))
        return cls(**d)


def ensure_compatible(a: IndexProfile | str, b: IndexProfile | str) -> None:
    ka = a if isinstance(a, str) else a.key
    kb = b if isinstance(b, str) else b.key
    if ka != kb:
        raise ProfileMismatch(f"index profile mismatch: {ka} != {kb}; reindex required")
