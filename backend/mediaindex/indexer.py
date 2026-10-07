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


LONG_AUDIO_S = 600.0  # decode per window above this, to bound memory


def make_audio_embedder(store: Store, host: ModelHost, batch_size: int = 4):
    """Embed fixed audio windows (mono 16 kHz) with the model's native audio encoder.

    All windows of a file are written in one transaction, so a cancelled/failed file never
    becomes partially searchable. Identical content reuses vectors and segment offsets.
    """
    from .media.audio import SAMPLE_RATE, decode_audio, plan_windows

    profile = host.profile
    profile_key = profile.key

    def embed_audio(item, ctx) -> dict:
        reused = store.reusable_segments(item.content_hash, profile_key, "audio")
        if reused is not None:
            vecs, segs = reused
            store.write_embeddings(item.asset_id, profile_key, "audio", vecs, segments=segs,
                                   source_hash=item.content_hash)
            return {"reused": 1, "segments": len(segs)}
        windows = plan_windows(item.info.duration, profile.audio_window_s, profile.audio_stride_s)
        whole = decode_audio(item.path) if item.info.duration <= LONG_AUDIO_S else None
        vectors = []
        for b in range(0, len(windows), batch_size):
            ctx.check_cancelled()
            chunk = windows[b:b + batch_size]
            arrays = []
            for s0, e0 in chunk:
                if whole is not None:
                    x = whole[int(s0 * SAMPLE_RATE): int(e0 * SAMPLE_RATE)]
                else:
                    x = decode_audio(item.path, start=s0, duration=e0 - s0)
                if x.size < SAMPLE_RATE // 10:  # <0.1 s: pad so the encoder gets a valid input
                    x = np.pad(x, (0, SAMPLE_RATE // 10 - x.size))
                arrays.append(x)
            with host.use() as backend:
                vectors.append(backend.embed_audio(arrays))
        mat = np.vstack(vectors)
        store.write_embeddings(item.asset_id, profile_key, "audio", mat, segments=windows,
                               source_hash=item.content_hash)
        return {"embedded": 1, "segments": len(windows)}

    embed_audio.profile_key = profile_key
    embed_audio.modality = "audio"
    return embed_audio
