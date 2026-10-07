"""Embedding backends.

`GemmaBackend` wraps the verified sentence-transformers interface for google/embeddinggemma-2
(see docs/model-compatibility.md). `FakeBackend` is a deterministic stand-in for unit tests only;
it is never selected at runtime unless MEDIAINDEX_FAKE_MODEL=1 is set explicitly for testing.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .profiles import IndexProfile


class ResourceError(RuntimeError):
    """Inference failed for resource reasons (e.g. out of memory)."""


class CapabilityUnavailable(RuntimeError):
    """The requested input mode is not supported by the loaded backend."""


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (n == 0).any():
        raise ValueError("non-finite or zero embedding produced")
    return x / n


class EmbeddingBackend(Protocol):
    profile: IndexProfile

    def capabilities(self) -> dict[str, bool]: ...
    def embed_query_texts(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def embed_image_text(self, image: Image.Image, text: str) -> np.ndarray: ...


def _is_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "mps backend out of memory" in s


class GemmaBackend:
    """Real EmbeddingGemma 2 inference. One instance per process."""

    def __init__(self, profile: IndexProfile, device: str = "auto", offline: bool = True):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.profile = profile
        config_kwargs = {}
        if "image" not in profile.encoders and "video" not in profile.encoders:
            config_kwargs["vision_config"] = None
        if "audio" not in profile.encoders:
            config_kwargs["audio_config"] = None
        self._torch = torch
        self.model = SentenceTransformer(
            profile.model_id,
            revision=profile.revision,
            device=device,
            model_kwargs={"dtype": getattr(torch, profile.precision)},
            config_kwargs=config_kwargs,
            local_files_only=offline,
        )
        self.model.eval()

    def capabilities(self) -> dict[str, bool]:
        enc = set(self.profile.encoders)
        return {
            "text": True,
            "image": "image" in enc,
            "image+text": "image" in enc,
            "audio": "audio" in enc,
            "video": "video" in enc,
        }

    def _encode(self, inputs, **kw) -> np.ndarray:
        try:
            with self._torch.inference_mode():
                out = self.model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True,
                                        batch_size=kw.pop("batch_size", 4), **kw)
        except RuntimeError as e:
            if _is_oom(e):
                self._empty_cache()
                raise ResourceError(f"out of memory on {self.device}: reduce batch size or use CPU") from e
            raise
        out = np.asarray(out, dtype=np.float32)
        if self.profile.dim < out.shape[-1]:
            out = out[..., : self.profile.dim]
        return l2_normalize(out)

    def _empty_cache(self) -> None:
        if self.device == "mps":
            self._torch.mps.empty_cache()

    def embed_query_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.profile.query_prompt + t for t in texts])

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not self.capabilities()["image"]:
            raise CapabilityUnavailable("image encoder not loaded in this profile")
        return self._encode([{"image": im.convert("RGB")} for im in images], batch_size=2)

    def embed_image_text(self, image: Image.Image, text: str) -> np.ndarray:
        """Native single-pass image+text embedding (verified interface, not an average)."""
        if not self.capabilities()["image+text"]:
            raise CapabilityUnavailable("image+text requires the image encoder")
        return self._encode([{"text": self.profile.query_prompt + text, "image": image.convert("RGB")}], batch_size=1)

    def close(self) -> None:
        del self.model
        self._empty_cache()


class FakeBackend:
    """Deterministic hash-based vectors. FOR UNIT TESTS ONLY; never real search."""

    def __init__(self, profile: IndexProfile, **_):
        self.profile = profile
        self.calls = 0

    def capabilities(self) -> dict[str, bool]:
        return {"text": True, "image": True, "image+text": True, "audio": "audio" in self.profile.encoders,
                "video": "video" in self.profile.encoders}

    def _vec(self, key: bytes) -> np.ndarray:
        self.calls += 1
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        return l2_normalize(np.random.default_rng(seed).standard_normal(self.profile.dim))

    def embed_query_texts(self, texts):
        return np.vstack([self._vec(b"t:" + t.encode()) for t in texts])

    def embed_images(self, images):
        return np.vstack([self._vec(b"i:" + im.convert("RGB").resize((16, 16)).tobytes()) for im in images])

    def embed_image_text(self, image, text):
        return self._vec(b"m:" + image.convert("RGB").resize((16, 16)).tobytes() + text.encode())

    def close(self) -> None:
        pass


def make_backend(profile: IndexProfile, device: str = "auto") -> EmbeddingBackend:
    if os.environ.get("MEDIAINDEX_FAKE_MODEL") == "1":
        return FakeBackend(profile)
    return GemmaBackend(profile, device=device, offline=os.environ.get("MEDIAINDEX_ALLOW_DOWNLOAD") != "1")
