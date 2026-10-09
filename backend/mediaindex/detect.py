"""Object counts for the Ask assistant: RT-DETR v2 (Apache-2.0, 80 COCO object types), run once per image.

Counting rule, the same for all 80 types, chosen from a measurement on 300 human-labelled COCO val2017 photos
(scripts/coco_check.py, docs/ask.md):
- boxes with score >= 0.75 are counted ("sure"): 96% of photos reported this way contained the thing;
- boxes with score 0.5-0.75 are kept as "maybe" and shown separately: only about 60% of them were right.
A chat-model double-check was measured too and helped little, so it is not used for counting.
Counts can still be wrong (small or crowded objects are missed, look-alikes are confused), so every answer shows the photos.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Protocol

from .jobs import JobContext
from .media.images import ImageRejected, load_image
from .paths import PathRejected, resolve_in_root
from .store import INDEXED, Store

DETECTOR_ID = "PekingU/rtdetr_v2_r50vd"
DETECTOR_REVISION = "282494075698cab9faa1096ae26856890030c817"
SCORE_THRESHOLD = 0.5  # boxes kept at all (below: "maybe")
SURE = 0.75  # boxes counted
DETECTOR_KEY = f"rtdetr_v2_r50vd@{DETECTOR_REVISION[:12]}:t{SCORE_THRESHOLD}:boxes"

# The detector's own label names, as written in its config.
LABELS = tuple(s.replace("_", " ") for s in (
    "aeroplane apple backpack banana baseball_bat baseball_glove bear bed bench bicycle bird boat book bottle bowl "
    "broccoli bus cake car carrot cat cell_phone chair clock cow cup diningtable dog donut elephant fire_hydrant fork "
    "frisbee giraffe hair_drier handbag horse hot_dog keyboard kite knife laptop microwave motorbike mouse orange "
    "oven parking_meter person pizza pottedplant refrigerator remote sandwich scissors sheep sink skateboard skis "
    "snowboard sofa spoon sports_ball stop_sign suitcase surfboard teddy_bear tennis_racket tie toaster toilet "
    "toothbrush traffic_light train truck tvmonitor umbrella vase wine_glass zebra").split())


Box = tuple[float, list[float]]  # (score, [x0, y0, x1, y1] as fractions of the image width and height)


class Detector(Protocol):
    def detect(self, image) -> dict[str, list[Box]]: ...  # label -> boxes with score >= SCORE_THRESHOLD
    def close(self) -> None: ...


class RTDetrDetector:
    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self._torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(DETECTOR_ID, revision=DETECTOR_REVISION)
        self.model = AutoModelForObjectDetection.from_pretrained(DETECTOR_ID, revision=DETECTOR_REVISION)
        self.model.to(device).eval()

    def detect(self, image, threshold: float = SCORE_THRESHOLD) -> dict[str, list[Box]]:
        torch = self._torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        r = self.processor.post_process_object_detection(out, target_sizes=torch.tensor([image.size[::-1]]),
                                                         threshold=threshold)[0]
        names = self.model.config.id2label
        w, h = image.size
        out: dict[str, list[Box]] = {}
        for i, sc, b in zip(r["labels"], r["scores"], r["boxes"]):
            x0, y0, x1, y1 = (float(v) for v in b)
            box = [round(max(0.0, min(1.0, v)), 4) for v in (x0 / w, y0 / h, x1 / w, y1 / h)]
            out.setdefault(names[int(i)], []).append((round(float(sc), 3), box))
        return out

    def close(self) -> None:
        self.model = None
        if self.device == "mps":
            self._torch.mps.empty_cache()


class DetectorHost:
    """Loads the detector on first use and frees it after each counting job (memory is tight on 8 GB Macs)."""

    def __init__(self, factory: Callable[[], Detector]):
        self._factory = factory
        self._detector: Detector | None = None
        self._lock = threading.Lock()

    def get(self) -> Detector:
        with self._lock:
            if self._detector is None:
                self._detector = self._factory()
            return self._detector

    def unload(self) -> None:
        with self._lock:
            if self._detector is not None:
                self._detector.close()
                self._detector = None


def run_detection(ctx: JobContext, store: Store, host: DetectorHost, library_ids: list[str] | None,
                  key: str = DETECTOR_KEY) -> dict:
    """Count objects in every indexed image that has not been checked yet. Read-only for the originals."""
    rows = images_in_scope(store, library_ids)
    done_hashes = {r["content_hash"] for r in store.db.query("SELECT content_hash FROM detection_runs WHERE detector=?",
                                                             (key,))}
    todo: dict[str, dict] = {}
    for r in rows:
        if r["content_hash"] not in done_hashes:
            todo.setdefault(r["content_hash"], r)
    ctx.progress(0, len(todo), "Counting objects")
    checked = failed = 0
    roots = {}
    try:
        for i, (h, r) in enumerate(todo.items(), 1):
            ctx.check_cancelled()
            root = roots.setdefault(r["library_id"], store.get_library(r["library_id"])["root_path"])
            try:
                im = load_image(resolve_in_root(root, r["rel_path"]))
                found = host.get().detect(im)
            except (ImageRejected, PathRejected, OSError):
                failed += 1
            else:
                store.save_detections(h, key, {k: sorted(v, key=lambda b: -b[0]) for k, v in found.items()})
                checked += 1
            ctx.progress(i, message=f"Counting objects: {i} of {len(todo)} photos")
    finally:
        host.unload()
    return {"checked": checked, "failed": failed,
            "already_checked": len({r["content_hash"] for r in rows}) - len(todo)}


def images_in_scope(store: Store, library_ids: list[str] | None) -> list[dict]:
    sql = "SELECT id, library_id, rel_path, content_hash FROM assets WHERE media_type='image' AND status=?"
    params: list = [INDEXED]
    if library_ids:
        sql += f" AND library_id IN ({','.join('?' * len(library_ids))})"
        params += library_ids
    return [dict(r) for r in store.db.query(sql + " ORDER BY rel_path", params)]


def default_factory(device: str) -> Callable[[], Detector]:
    return lambda: RTDetrDetector(device)


def detector_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(DETECTOR_ID, "config.json", revision=DETECTOR_REVISION)
    return isinstance(p, str) and Path(p).exists()
