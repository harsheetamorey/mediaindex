"""Search API. Similarity values are raw cosine scores, not probabilities or confidence."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .media.images import IMAGE_EXTENSIONS, ImageRejected, load_image
from .uploads import temp_upload
from .model.backend import CapabilityUnavailable, ResourceError
from .model.host import ModelBusy
from .search import MatrixCache, rank
from .store import index_state

MEDIA_MODALITIES = {"image": ["image"], "audio": ["audio"]}
SIMILARITY_NOTE = ("similarity is raw cosine similarity between embeddings; it is not a probability or confidence, "
                   "and nearest-neighbour search always returns candidates even when nothing truly matches")


class TextSearchRequest(BaseModel):
    mode: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=2000)
    library_ids: list[str] | None = None
    media_types: list[Literal["image", "audio"]] = ["image"]
    limit: int = Field(24, ge=1, le=200)
    group_segments: bool = True


def embed_query(app: FastAPI, fn):
    host = app.state.host
    try:
        with host.use(timeout=app.state.settings.query_wait_seconds) as backend:
            t0 = time.perf_counter()
            vec = fn(backend)
            return vec, (time.perf_counter() - t0) * 1000
    except ModelBusy as e:
        raise HTTPException(503, str(e))
    except CapabilityUnavailable as e:
        raise HTTPException(422, str(e))
    except ResourceError as e:
        raise HTTPException(507, str(e))
    except (OSError, RuntimeError) as e:
        if not host.loaded:
            raise HTTPException(503, f"model could not be loaded: {e}. Run the Phase 1 setup to cache weights.")
        raise


def group_hits(hits, limit: int, max_extra: int = 5):
    """Keep the best window per asset; attach up to `max_extra` other matching windows of that asset."""
    best: dict[str, dict] = {}
    order: list[str] = []
    for h in hits:
        g = best.get(h.asset_id)
        if g is None:
            if len(order) >= limit:
                continue
            best[h.asset_id] = {"hit": h, "others": []}
            order.append(h.asset_id)
        elif len(g["others"]) < max_extra:
            g["others"].append(h)
    return [best[a] for a in order]


def run_search(app: FastAPI, *, mode: str, query_info: dict, vec, embed_ms: float, library_ids, media_types,
               limit: int, exclude_assets: set[str] | None = None, group_segments: bool = True) -> dict:
    store = app.state.store
    profile_key = app.state.profile.key
    if library_ids:
        for lid in library_ids:
            if not store.get_library(lid):
                raise HTTPException(404, f"library not found: {lid}")
    state = index_state(store, profile_key, library_ids, list(media_types))
    if state["state"] == "incompatible":
        raise HTTPException(409, "the selected libraries were indexed with a different model profile; re-import to "
                                 "reindex them with the current profile")
    modalities = [m for t in media_types for m in MEDIA_MODALITIES[t]]
    cache: MatrixCache = app.state.matrix_cache
    t0 = time.perf_counter()
    vm = cache.get(profile_key, library_ids, modalities)
    segmented = any(m != "image" for m in modalities)
    if group_segments and segmented:
        groups = group_hits(rank(vec, vm, max(limit * 12, 200), exclude_assets), limit)
    else:
        groups = [{"hit": h, "others": []} for h in rank(vec, vm, limit, exclude_assets)]
    rank_ms = (time.perf_counter() - t0) * 1000
    results = []
    for g in groups:
        h = g["hit"]
        a = store.get_asset(h.asset_id)
        if a is None:  # removed between ranking and lookup
            continue
        results.append({"rank": len(results) + 1, "similarity": round(h.similarity, 5), "modality": h.modality,
                        "start_s": h.start_s, "end_s": h.end_s, "asset": app.state.public_asset(a),
                        "other_segments": [{"start_s": o.start_s, "end_s": o.end_s,
                                            "similarity": round(o.similarity, 5)} for o in g["others"]]})
    return {
        "mode": mode,
        "query": query_info,
        "profile_key": profile_key,
        "results": results,
        "candidates_searched": len(vm.asset_ids),
        "timing_ms": {"query_embedding": round(embed_ms, 2), "ranking": round(rank_ms, 3)},
        "index_state": state,
        "note": SIMILARITY_NOTE,
    }


def register(app: FastAPI) -> None:
    register_reference(app)
    register_mixed(app)
    register_audio(app)

    @app.post("/api/search/text")
    def search_text(req: TextSearchRequest) -> dict:
        vec, ms = embed_query(app, lambda b: b.embed_query_texts([req.text.strip()])[0])
        return run_search(app, mode="text", query_info={"text": req.text}, vec=vec, embed_ms=ms,
                          library_ids=req.library_ids, media_types=req.media_types, limit=req.limit,
                          group_segments=req.group_segments)


def _parse_libs(library_ids: str | None) -> list[str] | None:
    if not library_ids:
        return None
    return [x for x in (s.strip() for s in library_ids.split(",")) if x]


def _reference(app: FastAPI, file: UploadFile | None, asset_id: str | None):
    """Context-managed reference resolution. Yields (image_or_None, content_hash, asset_or_None, info)."""
    from contextlib import contextmanager

    @contextmanager
    def cm():
        if (file is None) == (asset_id is None):
            raise HTTPException(422, "provide exactly one of: an uploaded reference file, or asset_id")
        if file is not None:
            with temp_upload(file, app.state.settings.uploads_dir, IMAGE_EXTENSIONS) as (path, sha):
                try:
                    im = load_image(path)
                except ImageRejected as e:
                    reason = str(e).split(":")[0]  # never echo the private temp path
                    raise HTTPException(422, f"invalid reference image ({reason}); use a JPEG, PNG or WebP file")
                yield im, sha, None, {"reference": "upload", "filename": file.filename}
            return
        a = app.state.store.get_asset(asset_id)
        if a is None:
            raise HTTPException(404, "reference asset not found")
        if a["media_type"] != "image":
            raise HTTPException(422, "reference asset is not an image")
        _, path = app.state.asset_path(asset_id)  # ID-based, root-restricted
        try:
            im = load_image(path)
        except ImageRejected as e:
            raise HTTPException(422, f"invalid reference image: {e}")
        yield im, a["content_hash"], a, {"reference": "asset", "asset_id": asset_id, "rel_path": a["rel_path"]}

    return cm()


def _exclusions(app: FastAPI, ref_asset: dict | None, content_hash: str | None, include_identical: bool) -> set[str]:
    if include_identical:
        return set()
    ex = {ref_asset["id"]} if ref_asset else set()
    if content_hash:
        ex |= {a["id"] for a in app.state.store.find_by_hash(content_hash)}
    return ex


def register_reference(app: FastAPI) -> None:
    @app.post("/api/search/image")
    def search_image(file: UploadFile | None = File(None), asset_id: str | None = Form(None),
                     library_ids: str | None = Form(None), limit: int = Form(24, ge=1, le=200),
                     include_identical: bool = Form(False)) -> dict:
        with _reference(app, file, asset_id) as (im, sha, ref_asset, info):
            stored = None
            if ref_asset is not None:
                stored = app.state.store.get_asset_vectors(ref_asset["id"], app.state.profile.key, "image")
            if stored is not None:
                vec, ms = stored[0], 0.0
                info["embedding"] = "stored library vector"
            else:
                vec, ms = embed_query(app, lambda b: b.embed_images([im])[0])
                info["embedding"] = "computed"
        excl = _exclusions(app, ref_asset, sha, include_identical)
        info["excluded_identical"] = len(excl)
        return run_search(app, mode="image", query_info=info, vec=vec, embed_ms=ms,
                          library_ids=_parse_libs(library_ids), media_types=["image"], limit=limit,
                          exclude_assets=excl)


MIXED_INTERFACE = ("single forward pass: SentenceTransformer.encode({'text': query_prompt + text, 'image': <PIL image>}); "
                   "not an average of separate text and image vectors")


def register_mixed(app: FastAPI) -> None:
    @app.post("/api/search/image-text")
    def search_image_text(text: str = Form(..., min_length=1, max_length=2000),
                          file: UploadFile | None = File(None), asset_id: str | None = Form(None),
                          library_ids: str | None = Form(None), limit: int = Form(24, ge=1, le=200),
                          include_identical: bool = Form(False)) -> dict:
        text = text.strip()
        if not text:
            raise HTTPException(422, "refinement text is empty; use /api/search/image for reference-only search")
        caps = app.state.host.capabilities()
        if caps is not None and not caps.get("image+text"):
            raise HTTPException(422, "native image+text queries are not available with the loaded encoders")
        with _reference(app, file, asset_id) as (im, sha, ref_asset, info):
            vec, ms = embed_query(app, lambda b: b.embed_image_text(im, text)[0])
        profile = app.state.profile
        info.update({
            "text": text,
            "embedding": "computed (native image+text)",
            "interface": MIXED_INTERFACE,
            "model": profile.model_id,
            "revision": profile.revision,
            "prompt": profile.query_prompt,
            "caveat": ("refinement text steers the embedding; it does not enforce logical constraints, negation "
                       "or exact attributes"),
        })
        excl = _exclusions(app, ref_asset, sha, include_identical)
        info["excluded_identical"] = len(excl)
        return run_search(app, mode="image+text", query_info=info, vec=vec, embed_ms=ms,
                          library_ids=_parse_libs(library_ids), media_types=["image"], limit=limit,
                          exclude_assets=excl)


# ---- audio reference search (Phase 12) ------------------------------------------------
MAX_AUDIO_UPLOAD = 50 * 1024 * 1024


def _audio_reference(app: FastAPI, file: UploadFile | None, asset_id: str | None, start_s: float | None):
    """Yields (vector_or_None, decoded_array_or_None, content_hash, ref_asset, info)."""
    from contextlib import contextmanager

    from .media.audio import AUDIO_EXTENSIONS, SAMPLE_RATE, AudioRejected, decode_audio, probe_audio

    profile = app.state.profile

    @contextmanager
    def cm():
        if (file is None) == (asset_id is None):
            raise HTTPException(422, "provide exactly one of: an uploaded reference sound, or asset_id")
        if file is not None:
            with temp_upload(file, app.state.settings.uploads_dir, AUDIO_EXTENSIONS, MAX_AUDIO_UPLOAD) as (path, sha):
                try:
                    info = probe_audio(path)
                    s0 = max(0.0, min(start_s or 0.0, max(0.0, info.duration - 0.1)))
                    dur = min(profile.audio_window_s, info.duration - s0)
                    x = decode_audio(path, start=s0, duration=dur)
                except AudioRejected as e:
                    msg = str(e)
                    if "not found" in msg:
                        raise HTTPException(503, msg)
                    raise HTTPException(422, f"invalid reference sound ({msg.split(':')[0]}); use WAV, FLAC or MP3")
                if x.size < SAMPLE_RATE // 10:
                    x = np.pad(x, (0, SAMPLE_RATE // 10 - x.size))
                yield None, x, sha, None, {"reference": "upload", "filename": file.filename,
                                           "reference_span": [round(s0, 3), round(s0 + dur, 3)],
                                           "duration": round(info.duration, 3)}
            return
        a = app.state.store.get_asset(asset_id)
        if a is None:
            raise HTTPException(404, "reference asset not found")
        if a["media_type"] != "audio":
            raise HTTPException(422, "reference asset is not a sound")
        rows = app.state.store.db.query(
            """SELECT start_s, end_s, dim, vector FROM embeddings WHERE asset_id=? AND profile_key=? AND modality='audio'
               ORDER BY segment_index""", (asset_id, profile.key))
        if not rows:
            raise HTTPException(409, "reference sound is not indexed with the current profile; re-import its library")
        pick = rows[0]
        if start_s is not None:  # the window containing start_s (closest start otherwise)
            pick = min(rows, key=lambda r: (not (r["start_s"] <= start_s < r["end_s"]), abs(r["start_s"] - start_s)))
        from .store import blob_to_vec

        yield blob_to_vec(pick["vector"], pick["dim"]), None, a["content_hash"], a, {
            "reference": "asset", "asset_id": asset_id, "rel_path": a["rel_path"],
            "reference_span": [pick["start_s"], pick["end_s"]], "embedding": "stored segment vector"}

    return cm()


def register_audio(app: FastAPI) -> None:
    @app.post("/api/search/audio")
    def search_audio(file: UploadFile | None = File(None), asset_id: str | None = Form(None),
                     start_s: float | None = Form(None, ge=0), library_ids: str | None = Form(None),
                     limit: int = Form(24, ge=1, le=200), include_identical: bool = Form(False),
                     target: Literal["audio"] = Form("audio")) -> dict:
        with _audio_reference(app, file, asset_id, start_s) as (vec, x, sha, ref_asset, info):
            if vec is None:
                vec, ms = embed_query(app, lambda b: b.embed_audio([x])[0])
                info["embedding"] = "computed (audio encoder, mono 16 kHz)"
            else:
                ms = 0.0
        excl = _exclusions(app, ref_asset, sha, include_identical)
        info["excluded_identical"] = len(excl)
        return run_search(app, mode="audio", query_info=info, vec=vec, embed_ms=ms,
                          library_ids=_parse_libs(library_ids), media_types=[target], limit=limit,
                          exclude_assets=excl)
