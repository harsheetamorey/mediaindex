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
    global_rank: bool = False  # experimental: one list across media types by raw cosine


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


GLOBAL_RANK_NOTE = ("EXPERIMENTAL global ranking: images and sound windows are merged by raw cosine similarity. "
                    "Similarity scales are not calibrated across media types, so this order can favour one type.")


def run_search(app: FastAPI, *, mode: str, query_info: dict, vec, embed_ms: float, library_ids, media_types,
               limit: int, exclude_assets: set[str] | None = None, group_segments: bool = True,
               global_rank: bool = False) -> dict:
    media_types = list(dict.fromkeys(media_types))
    if len(media_types) > 1:
        parts = [_search_one(app, mode=mode, query_info=query_info, vec=vec, embed_ms=embed_ms,
                             library_ids=library_ids, media_types=[t], limit=limit, exclude_assets=exclude_assets,
                             group_segments=group_segments) for t in media_types]
        out = dict(parts[0])
        out["candidates_searched"] = sum(p["candidates_searched"] for p in parts)
        out["timing_ms"] = {"query_embedding": parts[0]["timing_ms"]["query_embedding"],
                            "ranking": round(sum(p["timing_ms"]["ranking"] for p in parts), 3)}
        out["index_state"] = {t: p["index_state"] for t, p in zip(media_types, parts)}
        if global_rank:
            merged = sorted((r for p in parts for r in p["results"]), key=lambda r: -r["similarity"])[:limit]
            for i, r in enumerate(merged):
                r["rank"] = i + 1
            out["results"] = merged
            out["grouping"] = "global (experimental)"
            out["note"] = SIMILARITY_NOTE + ". " + GLOBAL_RANK_NOTE
        else:
            out["results"] = [r for p in parts for r in p["results"]]
            out["groups"] = {t: len(p["results"]) for t, p in zip(media_types, parts)}
            out["grouping"] = "by_modality"
        out["media_types"] = media_types
        return out
    return _search_one(app, mode=mode, query_info=query_info, vec=vec, embed_ms=embed_ms, library_ids=library_ids,
                       media_types=media_types, limit=limit, exclude_assets=exclude_assets,
                       group_segments=group_segments)


def _search_one(app: FastAPI, *, mode: str, query_info: dict, vec, embed_ms: float, library_ids, media_types,
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
        "media_types": list(media_types),
        "grouping": "single",
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
    register_modes(app)

    @app.post("/api/search/text")
    def search_text(req: TextSearchRequest) -> dict:
        vec, ms = embed_query(app, lambda b: b.embed_query_texts([req.text.strip()])[0])
        return run_search(app, mode="text", query_info={"text": req.text}, vec=vec, embed_ms=ms,
                          library_ids=req.library_ids, media_types=req.media_types, limit=req.limit,
                          group_segments=req.group_segments, global_rank=req.global_rank)


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
                     include_identical: bool = Form(False),
                     target: Literal["image", "audio"] = Form("image")) -> dict:
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
        _mark_cross(info, "image", target)
        return run_search(app, mode="image" if target == "image" else "image→audio", query_info=info, vec=vec,
                          embed_ms=ms, library_ids=_parse_libs(library_ids), media_types=[target], limit=limit,
                          exclude_assets=excl)


MIXED_INTERFACE = ("single forward pass: SentenceTransformer.encode({'text': query_prompt + text, 'image': <PIL image>}); "
                   "not an average of separate text and image vectors")


def register_mixed(app: FastAPI) -> None:
    @app.post("/api/search/image-text")
    def search_image_text(text: str = Form(..., min_length=1, max_length=2000),
                          file: UploadFile | None = File(None), asset_id: str | None = Form(None),
                          library_ids: str | None = Form(None), limit: int = Form(24, ge=1, le=200),
                          include_identical: bool = Form(False),
                          target: Literal["image", "audio"] = Form("image")) -> dict:
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
        _mark_cross(info, "image+text", target)
        return run_search(app, mode="image+text" if target == "image" else "image+text→audio", query_info=info,
                          vec=vec, embed_ms=ms, library_ids=_parse_libs(library_ids), media_types=[target],
                          limit=limit, exclude_assets=excl)


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
                     target: Literal["audio", "image"] = Form("audio"),
                     text: str | None = Form(None, max_length=2000)) -> dict:
        text = (text or "").strip() or None
        with _audio_reference(app, file, asset_id, start_s) as (vec, x, sha, ref_asset, info):
            if text is not None:
                if x is None:  # asset reference: re-decode exactly the chosen window
                    x = _decode_ref_window(app, ref_asset, info["reference_span"])
                vec, ms = embed_query(app, lambda b: b.embed_audio_text(x, text)[0])
                info.update({"text": text, "embedding": "computed (native audio+text, single forward pass)",
                             "caveat": "EXPERIMENTAL: refinement text steers the embedding; no logical constraints"})
            elif vec is None:
                vec, ms = embed_query(app, lambda b: b.embed_audio([x])[0])
                info["embedding"] = "computed (audio encoder, mono 16 kHz)"
            else:
                ms = 0.0
        excl = _exclusions(app, ref_asset, sha, include_identical)
        info["excluded_identical"] = len(excl)
        _mark_cross(info, "audio+text" if text else "audio", target)
        mode = ("audio+text" if text else "audio") + ("" if target == "audio" else "→image")
        return run_search(app, mode=mode, query_info=info, vec=vec, embed_ms=ms,
                          library_ids=_parse_libs(library_ids), media_types=[target], limit=limit,
                          exclude_assets=excl)


def _decode_ref_window(app: FastAPI, asset: dict, span) -> np.ndarray:
    from .media.audio import AudioRejected, decode_audio

    _, path = app.state.asset_path(asset["id"])
    try:
        return decode_audio(path, start=span[0], duration=span[1] - span[0])
    except AudioRejected as e:
        raise HTTPException(422, f"cannot decode reference sound: {e}")


# Supported query modes. "verified": exercised with real data in an earlier phase; "experimental":
# runs on the shared embedding space but quality is unmeasured/uneven (see docs/cross-modal-findings.md).
QUERY_MODES = [
    {"query": "text", "target": "image", "status": "verified", "endpoint": "/api/search/text", "needs": ["image"]},
    {"query": "text", "target": "audio", "status": "verified", "endpoint": "/api/search/text", "needs": ["audio"]},
    {"query": "text", "target": "image+audio", "status": "verified (grouped by type)", "endpoint": "/api/search/text",
     "needs": ["image", "audio"]},
    {"query": "image", "target": "image", "status": "verified", "endpoint": "/api/search/image", "needs": ["image"]},
    {"query": "image+text", "target": "image", "status": "verified (quality caveats)", "endpoint": "/api/search/image-text",
     "needs": ["image"]},
    {"query": "audio", "target": "audio", "status": "verified", "endpoint": "/api/search/audio", "needs": ["audio"]},
    {"query": "image", "target": "audio", "status": "experimental", "endpoint": "/api/search/image", "needs": ["image", "audio"]},
    {"query": "audio", "target": "image", "status": "experimental", "endpoint": "/api/search/audio", "needs": ["image", "audio"]},
    {"query": "image+text", "target": "audio", "status": "experimental", "endpoint": "/api/search/image-text",
     "needs": ["image", "audio"]},
    {"query": "audio+text", "target": "audio", "status": "experimental", "endpoint": "/api/search/audio", "needs": ["audio"]},
    {"query": "audio+text", "target": "image", "status": "experimental", "endpoint": "/api/search/audio",
     "needs": ["image", "audio"]},
]


def _mark_cross(info: dict, query: str, target: str) -> None:
    m = next((x for x in QUERY_MODES if x["query"] == query and x["target"] == target), None)
    info["mode_status"] = m["status"] if m else "unsupported"
    if m and m["status"] == "experimental":
        info["experimental"] = True


def register_modes(app: FastAPI) -> None:
    @app.get("/api/capabilities")
    def capabilities() -> dict:
        enc = set(app.state.profile.encoders)
        modes = [dict(m, available=all(n in enc for n in m["needs"])) for m in QUERY_MODES]
        return {"encoders": sorted(enc), "modes": modes,
                "notes": ["Results are grouped by media type by default because raw similarity scales are not "
                          "calibrated across images and sounds.", GLOBAL_RANK_NOTE,
                          "A sound suggested for an image is a similarity candidate, not synchronization or a "
                          "judgement of artistic quality."]}
