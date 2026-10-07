"""Search API. Similarity values are raw cosine scores, not probabilities or confidence."""

from __future__ import annotations

import time
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .media.images import IMAGE_EXTENSIONS, ImageRejected, load_image
from .uploads import temp_upload
from .model.backend import CapabilityUnavailable, ResourceError
from .model.host import ModelBusy
from .search import MatrixCache, rank
from .store import index_state

MEDIA_MODALITIES = {"image": ["image"]}
SIMILARITY_NOTE = ("similarity is raw cosine similarity between embeddings; it is not a probability or confidence, "
                   "and nearest-neighbour search always returns candidates even when nothing truly matches")


class TextSearchRequest(BaseModel):
    mode: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=2000)
    library_ids: list[str] | None = None
    media_types: list[Literal["image"]] = ["image"]
    limit: int = Field(24, ge=1, le=200)


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


def run_search(app: FastAPI, *, mode: str, query_info: dict, vec, embed_ms: float, library_ids, media_types,
               limit: int, exclude_assets: set[str] | None = None) -> dict:
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
    hits = rank(vec, vm, limit, exclude_assets)
    rank_ms = (time.perf_counter() - t0) * 1000
    results = []
    for i, h in enumerate(hits):
        a = store.get_asset(h.asset_id)
        if a is None:  # removed between ranking and lookup
            continue
        results.append({"rank": i + 1, "similarity": round(h.similarity, 5), "modality": h.modality,
                        "start_s": h.start_s, "end_s": h.end_s, "asset": app.state.public_asset(a)})
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

    @app.post("/api/search/text")
    def search_text(req: TextSearchRequest) -> dict:
        vec, ms = embed_query(app, lambda b: b.embed_query_texts([req.text.strip()])[0])
        return run_search(app, mode="text", query_info={"text": req.text}, vec=vec, embed_ms=ms,
                          library_ids=req.library_ids, media_types=req.media_types, limit=req.limit)


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
                    raise HTTPException(422, f"invalid reference image: {e}")
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
