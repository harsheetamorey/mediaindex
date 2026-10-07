"""Exact cosine ranking over normalized vectors (dot product), with a generation-checked cache."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .store import Store, VectorMatrix


@dataclass
class Hit:
    embedding_id: str
    asset_id: str
    similarity: float
    start_s: float | None
    end_s: float | None
    modality: str


def rank(query: np.ndarray, vm: VectorMatrix, limit: int, exclude_assets: set[str] | None = None) -> list[Hit]:
    """Top-`limit` by dot product. Query and matrix rows must be L2-normalized."""
    if vm.matrix.size == 0:
        return []
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if q.shape[0] != vm.matrix.shape[1]:
        raise ValueError(f"query dim {q.shape[0]} != index dim {vm.matrix.shape[1]}")
    sims = vm.matrix @ q
    if exclude_assets:
        mask = np.fromiter((a in exclude_assets for a in vm.asset_ids), bool, len(vm.asset_ids))
        sims = np.where(mask, -np.inf, sims)
    n = int(np.isfinite(sims).sum())
    k = min(limit, n)
    if k <= 0:
        return []
    idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
    idx = idx[np.argsort(-sims[idx], kind="stable")]
    return [Hit(vm.embedding_ids[i], vm.asset_ids[i], float(sims[i]), vm.starts[i], vm.ends[i], vm.modalities[i])
            for i in idx if np.isfinite(sims[i])]


class MatrixCache:
    """Caches loaded matrices; invalidated whenever any library generation changes."""

    def __init__(self, store: Store):
        self.store = store
        self._cache: dict[tuple, tuple[tuple, VectorMatrix]] = {}
        self._lock = threading.Lock()

    def _generation(self) -> tuple:
        r = self.store.db.one("""SELECT COALESCE(SUM(generation),0) AS g, COUNT(*) AS n,
                                 (SELECT COUNT(*) FROM embeddings) AS e FROM libraries""")
        return (r["g"], r["n"], r["e"])

    def get(self, profile_key: str, library_ids: list[str] | None, modalities: list[str] | None) -> VectorMatrix:
        key = (profile_key, tuple(sorted(library_ids or [])), tuple(sorted(modalities or [])))
        gen = self._generation()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] == gen:
                return hit[1]
        vm = self.store.load_matrix(profile_key, library_ids, modalities)
        with self._lock:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = (gen, vm)
        return vm
