"""Connect ingestion to the model: embed pending images (reusing vectors for identical content)."""

from __future__ import annotations

import numpy as np

from .ingest import PendingImage
from .model.host import ModelHost
from .store import FAILED, Store


def make_image_embedder(store: Store, host: ModelHost):
    profile_key = host.profile.key

    def embed_batch(items: list[PendingImage]) -> dict:
        out = {"embedded": 0, "reused": 0, "failed": []}
        todo: list[PendingImage] = []
        for it in items:
            vec = store.reusable_vectors(it.content_hash, profile_key, "image")
            if vec is not None:
                store.write_embeddings(it.asset_id, profile_key, "image", vec, source_hash=it.content_hash)
                out["reused"] += 1
            else:
                todo.append(it)
        if not todo:
            return out
        # Hold the model lock only for this batch so queries can interleave between batches.
        try:
            with host.use() as backend:
                vecs = backend.embed_images([it.image for it in todo])
        except Exception as e:
            # Fall back to per-item so one bad image doesn't fail the batch.
            vecs = []
            for it in todo:
                try:
                    with host.use() as backend:
                        vecs.append(backend.embed_images([it.image])[0])
                except Exception as e2:  # noqa: BLE001
                    vecs.append(None)
                    store.set_asset_status(it.asset_id, FAILED, f"embedding failed: {e2}")
                    out["failed"].append((it.rel_path, f"embedding failed: {e2}"))
            if all(v is None for v in vecs):
                out.setdefault("batch_error", str(e))
        for it, v in zip(todo, vecs):
            if v is None:
                continue
            store.write_embeddings(it.asset_id, profile_key, "image", np.asarray(v)[None, :],
                                   source_hash=it.content_hash)
            out["embedded"] += 1
        return out

    embed_batch.profile_key = profile_key
    embed_batch.modality = "image"
    return embed_batch
