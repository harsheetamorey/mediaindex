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


def make_video_embedder(store: Store, host: ModelHost, thumbs_dir):
    """Index overlapping video windows: native video (frames at profile.video_fps), the window's
    audio track via the audio encoder, and optionally a joint video+audio embedding.

    Visual vectors come from the model's native video input over the sampled frames; they are NOT an
    average of per-frame image vectors. All windows/modalities of a file are written in one transaction.
    """
    from PIL import Image

    from .ingest import window_thumb_path
    from .media.audio import AudioRejected, decode_audio, plan_windows
    from .media.images import write_thumbnail
    from .media.video import sample_frames

    profile = host.profile
    key = profile.key
    mods = set(profile.video_modalities)

    def embed_video(item, ctx) -> dict:
        reused_any = False
        cached = {m: store.reusable_segments(item.content_hash, key, m) for m in mods}
        if cached.get("video-visual") is not None:
            with store.db.tx() as c:
                for m, r in cached.items():
                    if r is not None:
                        store.write_embeddings(item.asset_id, key, m, r[0], segments=r[1],
                                               source_hash=item.content_hash, c=c)
            return {"reused": 1, "windows": len(cached["video-visual"][1])}
        windows = plan_windows(item.info.duration, profile.video_window_s, profile.video_stride_s)
        out: dict[str, list] = {m: [] for m in mods}
        for s0, e0 in windows:
            ctx.check_cancelled()
            frames, times = sample_frames(item.path, item.info, s0, e0 - s0, fps=profile.video_fps)
            tp = window_thumb_path(thumbs_dir, item.content_hash, s0)
            if not tp.exists():
                write_thumbnail(Image.fromarray(frames[len(frames) // 2]), tp)
            wav = None
            if item.info.has_audio and ({"video-audio", "video-joint"} & mods):
                try:
                    wav = decode_audio(item.path, start=s0, duration=e0 - s0)
                except AudioRejected:
                    wav = None
            with host.use() as backend:
                if "video-visual" in mods:
                    out["video-visual"].append(backend.embed_video([frames], profile.video_fps)[0])
                if "video-audio" in mods and wav is not None:
                    out["video-audio"].append((s0, e0, backend.embed_audio([wav])[0]))
                if "video-joint" in mods and wav is not None:
                    out["video-joint"].append((s0, e0, backend.embed_video_audio(frames, profile.video_fps, wav)[0]))
        with store.db.tx() as c:
            if out.get("video-visual"):
                store.write_embeddings(item.asset_id, key, "video-visual", np.vstack(out["video-visual"]),
                                       segments=windows, source_hash=item.content_hash, mark_indexed=False, c=c)
            for m in ("video-audio", "video-joint"):
                if out.get(m):
                    segs = [(a, b) for a, b, _ in out[m]]
                    store.write_embeddings(item.asset_id, key, m, np.vstack([v for *_, v in out[m]]),
                                           segments=segs, source_hash=item.content_hash, mark_indexed=False, c=c)
            store.set_asset_status(item.asset_id, "indexed", None, c=c)
        return {"embedded": 1, "windows": len(windows), "reused": int(reused_any)}

    embed_video.profile_key = key
    embed_video.modality = "video-visual"
    return embed_video
