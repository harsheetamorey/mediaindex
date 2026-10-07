"""Search API. Similarity values are raw cosine scores, not probabilities or confidence."""

from __future__ import annotations

import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
    @app.post("/api/search/text")
    def search_text(req: TextSearchRequest) -> dict:
        vec, ms = embed_query(app, lambda b: b.embed_query_texts([req.text.strip()])[0])
        return run_search(app, mode="text", query_info={"text": req.text}, vec=vec, embed_ms=ms,
                          library_ids=req.library_ids, media_types=req.media_types, limit=req.limit)

