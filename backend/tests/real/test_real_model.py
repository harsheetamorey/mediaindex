"""REAL-MODEL checks (opt-in): `uv run pytest -m real_model`. Requires cached weights."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mediaindex.config import Settings
from mediaindex.model.backend import GemmaBackend
from mediaindex.model.host import ModelHost
from mediaindex.model.profiles import IndexProfile

pytestmark = pytest.mark.real_model
SAMPLES = Path(__file__).resolve().parents[3] / "data" / "samples" / "smoke"


@pytest.fixture(scope="module")
def host():
    s = Settings()
    h = ModelHost(IndexProfile(precision=s.resolved_precision()),
                  lambda p: GemmaBackend(p, device=s.resolved_device(), offline=True))
    yield h
    h.shutdown()


def test_real_text_image_mixed_and_reuse(host):
    imgs = [Image.open(p) for p in sorted(SAMPLES.glob("*.jpg"))[:3]]
    assert len(imgs) == 3, "run Phase 1 sample prep first (data/samples/smoke)"
    with host.use() as b:
        q = b.embed_query_texts(["a zebra", "a guitar"])
        d = b.embed_images(imgs)
        m = b.embed_image_text(imgs[0], "a musical instrument")
    with host.use() as b:
        q2 = b.embed_query_texts(["a zebra", "a guitar"])
    assert host.load_count == 1
    for arr in (q, d, m):
        assert np.isfinite(arr).all() and arr.shape[-1] == 768
        assert np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-5)
    assert np.allclose(q, q2, atol=1e-5)
