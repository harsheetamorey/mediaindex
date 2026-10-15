"""Photo edits for the Ask tab: remove an object from a photo, or replace it with something else.

- Which object: the detector's boxes (detect.py) for the thing the user names.
- Its exact outline: SAM (Segment Anything, ViT-B, Apache-2.0), prompted with each box.
- Remove: LaMa (big-lama, Apache-2.0) fills the outlined area from its surroundings.
- Replace: Stable Diffusion inpainting paints the new object into the outlined area (see creative.py).

Originals are never modified: every edit is a new JPEG under <data>/edits, served by its random id. Only the pixels
inside the (slightly grown) outline change; everything else is copied from the original unchanged.
Models load on first use and stay loaded for a few minutes (KeepWarm), so the next edit skips loading; they are freed
when idle, or right after an edit if memory is low, because memory is tight on 8 GB Macs.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageFilter

SAM_ID = "facebook/sam-vit-base"
SAM_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
LAMA_REPO = "fashn-ai/LaMa"  # the original big-lama weights (advimman/lama, Apache-2.0) as TorchScript
LAMA_REVISION = "da55aaf971ce271b42eddca79831e0f646c7711e"
LAMA_FILE = "big-lama.pt"
LAMA_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"
DOWNLOAD_HINT = "MEDIAINDEX_ALLOW_DOWNLOAD=1 uv run python -m mediaindex.edit --download"

GROW = 0.012  # the outline is grown by this fraction of the photo's longer side, so edges and shadows go too
FILL_MAX_SIDE = 1024  # LaMa works on a crop around the object, scaled to at most this size
JPEG_QUALITY = 95
REPLACE_MASK = "box"  # replacing paints the object's box (plus outline), not its silhouette: the new thing has its own shape
REPLACE_CONTEXT = 0.5  # and looks at less surroundings than removing, so the object is big enough on the model's canvas
KEEP_WARM_S = 300  # keep edit models loaded this long after the last edit (replacing: 89 s with loading, 35 s without)
MIN_AVAILABLE = 1.5 * 2**30  # but free them right after an edit when less memory than this is available
MAX_CHANGE = 0.35  # never change more than this share of a photo: a bigger hole can only be filled with made-up scenery
BIG_REMOVE = 0.12  # removing more than this share of a photo: LaMa smears, so the image model paints the background
BIG_FILL_SIDE = 512  # and LaMa (the fallback) fills the whole box at this size: fewer blocks than at 1024
REFINE_STRENGTH = 0.6  # second try: lightly repaint LaMa's fill (0.4 left a blur; full strength often adds a new car)
MAX_CHANGE_ASKED = 0.9  # unless the user says to go ahead anyway: then most of the photo may be made up


class EditRefused(ValueError):
    """The edit would not give a believable result (e.g. the object fills most of the photo); nothing was changed."""

PixelBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
Found = list[tuple[str, PixelBox]]  # (label, box) for every object the detector finds


class Segmenter(Protocol):
    def masks(self, image: Image.Image, boxes: list[PixelBox]) -> list[np.ndarray]: ...  # one HxW bool mask per box
    def close(self) -> None: ...


class Filler(Protocol):
    def fill(self, image: Image.Image, mask: np.ndarray) -> Image.Image: ...  # same size; only `mask` pixels change
    def close(self) -> None: ...


def _empty_cache(torch, device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()


class SamSegmenter:
    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import SamModel, SamProcessor

        self._torch = torch
        self.device = device
        self.processor = SamProcessor.from_pretrained(SAM_ID, revision=SAM_REVISION)
        self.model = SamModel.from_pretrained(SAM_ID, revision=SAM_REVISION).to(device).eval()

    def masks(self, image: Image.Image, boxes: list[PixelBox]) -> list[np.ndarray]:
        if not boxes:
            return []
        torch = self._torch
        image = image.convert("RGB")
        inputs = self.processor(image, input_boxes=[[list(b) for b in boxes]], return_tensors="pt")
        sizes, resized = inputs["original_sizes"], inputs["reshaped_input_sizes"]
        inputs = {k: (v.float() if v.dtype == torch.float64 else v).to(self.device)  # MPS has no float64
                  for k, v in inputs.items() if hasattr(v, "dtype")}
        with torch.inference_mode():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(out.pred_masks.cpu(), sizes, resized)[0]  # (n, 3, H, W)
        scores = out.iou_scores.cpu()[0]  # (n, 3)
        result = []
        for i in range(masks.shape[0]):
            # SAM offers 3 outlines (part, object, group). The detector's box says how big the object is, so pick
            # the outline whose extent matches the box best (SAM's own score alone often picks a part, e.g. a face).
            cands = [masks[i, j].numpy().astype(bool) for j in range(masks.shape[1])]
            best = max(range(len(cands)), key=lambda j: _box_iou(_extent(cands[j]), boxes[i]) + 0.25 * float(scores[i, j]))
            result.append(_keep_in_box(cands[best], boxes[i]))
        return result

    def close(self) -> None:
        self.model = None
        _empty_cache(self._torch, self.device)


def _extent(mask: np.ndarray) -> PixelBox:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _box_iou(a: PixelBox, b: PixelBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside(box: PixelBox, mask: np.ndarray, share: float = 0.5) -> bool:
    """At least `share` of the box is inside the mask."""
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    r = mask[y0:int(np.ceil(box[3])), x0:int(np.ceil(box[2]))]
    return bool(r.size) and float(r.mean()) >= share


def _with_boxes(raw: np.ndarray, boxes: list[PixelBox], px: int) -> np.ndarray:
    """The outline plus each object's whole box, grown by px: room for shadows and the bits the outline missed."""
    m = raw.copy()
    for b in boxes:
        m[max(0, int(b[1])):int(np.ceil(b[3])), max(0, int(b[0])):int(np.ceil(b[2]))] = True
    return grow(m, px)


def _keep_in_box(mask: np.ndarray, box: PixelBox, margin: float = 0.08) -> np.ndarray:
    """SAM sometimes spills onto neighbours of the same colour: keep the mask inside the (slightly larger) box."""
    h, w = mask.shape
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    keep = np.zeros_like(mask)
    keep[max(0, int(y0 - my)):min(h, int(np.ceil(y1 + my))), max(0, int(x0 - mx)):min(w, int(np.ceil(x1 + mx)))] = True
    return mask & keep


class LamaFiller:
    def __init__(self, device: str = "cpu"):
        import torch
        from huggingface_hub import hf_hub_download

        self._torch = torch
        path = hf_hub_download(LAMA_REPO, LAMA_FILE, revision=LAMA_REVISION)
        if _sha256(path) != LAMA_SHA256:
            raise RuntimeError(f"{LAMA_FILE} does not match its pinned checksum; delete it and download again")
        # LaMa's Fourier convolutions are not all supported on MPS; the CPU is fast enough at this size
        self.device = "cpu"
        self.model = torch.jit.load(path, map_location="cpu").eval()

    def fill(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        torch = self._torch
        img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        h, w = mask.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8  # LaMa needs sides divisible by 8
        img_t = torch.from_numpy(np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")).permute(2, 0, 1)[None]
        m_t = torch.from_numpy(np.pad(mask.astype(np.float32), ((0, ph), (0, pw)), mode="constant"))[None, None]
        with torch.inference_mode():
            out = self.model(img_t, m_t)[0].permute(1, 2, 0).numpy()[:h, :w]
        return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))

    def close(self) -> None:
        self.model = None


def _available_memory() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available)
    except Exception:  # can't tell: behave as if memory is low
        return 0.0


class KeepWarm:
    """A model factory that keeps the model it made for `idle` seconds after use. Callers use it like the factory:
    `m = warm(); try: ... finally: m.close()`; close() ends the use, and the model is really closed when idle, when
    memory is low, or when another kind (other args) is asked for. One use at a time."""

    def __init__(self, make: Callable, idle: float = KEEP_WARM_S, min_available: float = MIN_AVAILABLE,
                 available: Callable[[], float] = _available_memory):
        self._make, self.idle, self.min_available, self._available = make, idle, min_available, available
        self._model, self._args, self._last, self._timer = None, None, 0.0, None
        self._lock = threading.Lock()

    def __getattr__(self, name):  # model_name etc. of the wrapped factory
        return getattr(self._make, name)

    def __call__(self, *args):
        self._lock.acquire()
        try:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._model is not None and self._args != args:
                self._free()
            if self._model is None:
                self._model, self._args = self._make(*args), args
        except BaseException:
            self._lock.release()
            raise
        return _Lease(self, self._model)

    def _release(self) -> None:
        try:
            self._last = time.monotonic()
            if self.idle <= 0 or self._available() < self.min_available:
                self._free()
            else:
                self._timer = threading.Timer(self.idle, self._expire)
                self._timer.daemon = True
                self._timer.start()
        finally:
            self._lock.release()

    def _expire(self) -> None:
        with self._lock:
            if time.monotonic() - self._last >= self.idle - 0.5:  # not used again since this timer was set
                self._free()

    def _free(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._model is not None:
            model, self._model, self._args = self._model, None, None
            model.close()
            import gc

            gc.collect()

    def unload(self) -> None:
        with self._lock:
            self._free()

    @property
    def loaded(self) -> bool:
        return self._model is not None


class _Lease:
    """The warm model for one use: attributes go to the model; close() hands it back to KeepWarm."""

    def __init__(self, warm: KeepWarm, model):
        object.__setattr__(self, "_warm", warm)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_open", True)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        setattr(self._model, name, value)

    def close(self) -> None:
        if self._open:
            object.__setattr__(self, "_open", False)
            self._warm._release()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow a mask by about `px` pixels in every direction."""
    if px <= 0 or not mask.any():
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    small = max(1, int(np.ceil(px / 6)))  # MaxFilter is slow for big kernels: grow at a reduced scale, then scale back
    scale = max(1, px // 6)
    if scale > 1:
        sm = im.resize((max(1, im.width // scale), max(1, im.height // scale)), Image.Resampling.BOX)
        sm = sm.point(lambda v: 255 if v > 0 else 0).filter(ImageFilter.MaxFilter(2 * small + 1))
        big = sm.resize(im.size, Image.Resampling.NEAREST)
        return (np.asarray(big) > 0) | mask
    return np.asarray(im.filter(ImageFilter.MaxFilter(2 * px + 1))) > 0


def crop_box(mask: np.ndarray, context: float = 1.0) -> tuple[int, int, int, int]:
    """A region around the mask with `context` x its size of surroundings on each side (what the filler looks at)."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    pad = int(max(side * context, 0.08 * max(w, h)))
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def fill_region(image: Image.Image, mask: np.ndarray, fill: Callable[[Image.Image, np.ndarray], Image.Image],
                max_side: int | None = None, context: float = 1.0) -> Image.Image:
    """Run `fill` on a crop around the mask (scaled to max_side), then paste only the masked pixels back."""
    max_side = max_side or FILL_MAX_SIDE
    image = image.convert("RGB")
    x0, y0, x1, y1 = crop_box(mask, context)
    crop = image.crop((x0, y0, x1, y1))
    cm = mask[y0:y1, x0:x1]
    scale = min(1.0, max_side / max(crop.size))
    size = (max(8, round(crop.width * scale)), max(8, round(crop.height * scale)))
    small = crop.resize(size, Image.Resampling.LANCZOS) if scale < 1 else crop
    small_mask = np.asarray(Image.fromarray(cm.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 0 \
        if scale < 1 else cm
    filled = fill(small, grow(small_mask, 2) if scale < 1 else small_mask)
    if filled.size != crop.size:
        filled = filled.resize(crop.size, Image.Resampling.LANCZOS)
    alpha = Image.fromarray(cm.astype(np.uint8) * 255)
    out = image.copy()
    out.paste(Image.composite(filled, crop, alpha), (x0, y0))
    return out


@dataclass
class EditResult:
    image: Image.Image
    mask: np.ndarray  # the pixels that changed may only be inside this
    objects: int
    seconds: float
    attempts: int = 1
    clean: bool | None = None  # None: not checked; False: the object may still be visible
    new_found: bool | None = None  # replacing: None if not checked; False if a check doesn't find the new thing
    filled_by: str = ""  # removing: "lama", or "image model" when it painted the background of a big object


class Editor:
    """Gets SAM and the fillers from their factories for each edit and closes them after it (pass KeepWarm factories
    to keep them loaded between edits). One edit at a time."""

    def __init__(self, segmenter_factory: Callable[[], Segmenter], filler_factory: Callable[[], Filler],
                 painter_factory: Callable[[], object] | None = None):
        self._seg_factory = segmenter_factory
        self._fill_factory = filler_factory
        self._paint_factory = painter_factory
        self._lock = threading.Lock()

    def outline(self, image: Image.Image, boxes: list[PixelBox]) -> np.ndarray:
        """The pixels to change: the objects' outlines, grown a little."""
        return grow(self.segment(image, boxes), max(2, round(GROW * max(image.size))))

    def segment(self, image: Image.Image, boxes: list[PixelBox]) -> np.ndarray:
        """The objects' exact outlines (union of one SAM mask per box)."""
        seg = self._seg_factory()
        try:
            masks = seg.masks(image, boxes)
        finally:
            seg.close()
        mask = np.zeros((image.height, image.width), dtype=bool)
        for m, b in zip(masks, boxes):
            if m.sum() < 0.05 * max(1.0, (b[2] - b[0]) * (b[3] - b[1])):  # SAM found almost nothing: use the box
                m = np.zeros_like(mask)
                m[int(b[1]):int(np.ceil(b[3])), int(b[0]):int(np.ceil(b[2]))] = True
            mask |= m
        return mask

    def remove(self, image: Image.Image, boxes: list[PixelBox],
               still_there: Callable[[Image.Image], list[PixelBox]] | None = None,
               max_change: float = MAX_CHANGE, find_all: Callable[[Image.Image], Found] | None = None,
               seed: int = 0, place: str = "background") -> EditResult:
        """Fill the objects' outlines from their surroundings.

        still_there(edited) returns the boxes where the object is still detected. If any remain, the area is grown
        more (edges, shadows and leftovers go too) and filled again from the original, once; `clean` says whether the
        final check passed.

        Big objects (over BIG_REMOVE of the photo), when the image model and find_all are there: LaMa smears big holes,
        so the whole box is filled by the image model painting empty background instead, then by lightly repainting
        LaMa's fill. A try is kept only if find_all finds nothing new in the filled area (the model likes to paint a
        new car where a car was); otherwise LaMa's fill of the box is kept.
        """
        with self._lock:
            t = time.perf_counter()
            raw = self.segment(image, boxes)
            side = max(image.size)
            mask = grow(raw, max(2, round(GROW * side)))
            if mask.mean() > max_change:
                raise EditRefused(f"it covers about {round(100 * float(raw.mean()))}% of the photo")
            big = mask.mean() > BIG_REMOVE and find_all is not None
            if big:
                boxed = _with_boxes(raw, boxes, max(2, round(GROW * side)))
                if boxed.mean() <= max_change:
                    mask = boxed
            filler = self._fill_factory()
            try:
                out = fill_region(image, mask, filler.fill, max_side=BIG_FILL_SIDE if big else None)
                attempts, clean = 1, None
                if still_there is not None and not big:
                    left = [b for b in still_there(out) if any(_box_iou(b, x) >= 0.3 for x in boxes)]
                    clean = not left
                    wider = grow(raw, max(4, round(3 * GROW * side)))
                    for b in left:  # the whole box where it was detected again: the outline missed part of it
                        wider[int(b[1]):int(np.ceil(b[3])), int(b[0]):int(np.ceil(b[2]))] = True
                    if left and wider.mean() <= max_change:
                        mask = wider
                        out = fill_region(image, mask, filler.fill)
                        attempts = 2
                        clean = not [b for b in still_there(out) if any(_box_iou(b, x) >= 0.3 for x in boxes)]
            finally:
                filler.close()  # before the image model loads: never both in memory on an 8 GB Mac
            filled_by = "lama"
            if big:
                out, attempts, clean, filled_by = self._paint_background(image, out, mask, boxes, find_all, seed, place)
            return EditResult(out, mask, len(boxes), time.perf_counter() - t, attempts, clean, filled_by=filled_by)

    def _paint_background(self, image: Image.Image, lama: Image.Image, mask: np.ndarray, boxes: list[PixelBox],
                          find_all: Callable[[Image.Image], Found], seed: int,
                          place: str = "background") -> tuple[Image.Image, int, bool, str]:
        """Returns (image, tries, clean, filled_by). clean: nothing new or left over is found in the filled area."""
        kept = [(lab, b) for lab, b in find_all(image)  # other things there before, partly under the mask
                if _inside(b, mask) and not any(_box_iou(b, x) >= 0.3 for x in boxes)]

        def new_things(out: Image.Image) -> list[str]:
            return [lab for lab, b in find_all(out)
                    if _inside(b, mask) and not any(lab == k and _box_iou(b, kb) >= 0.5 for k, kb in kept)]

        lama_clean = not new_things(lama)
        if self._paint_factory is None:
            return lama, 1, lama_clean, "lama"
        try:
            painter = self._paint_factory()
        except Exception:  # not installed or not downloaded: LaMa's fill it is
            return lama, 1, lama_clean, "lama"
        start = getattr(painter, "seed", None)
        try:
            if start is not None:
                painter.seed = start + seed  # type: ignore[attr-defined]  # "try again" asks for another picture
            tries = [lambda im, m: painter.clear(im, m, place=place),  # type: ignore[attr-defined]
                     lambda im, m: painter.clear(im, m, strength=REFINE_STRENGTH, place=place)]  # type: ignore[attr-defined]
            for n, (paint, base) in enumerate(zip(tries, (image, lama)), 2):
                out = fill_region(base, mask, paint, max_side=painter.max_side, context=1.0)  # type: ignore[attr-defined]
                if not new_things(out):
                    return out, n, True, "image model"
        finally:
            if start is not None:
                painter.seed = start  # type: ignore[attr-defined]
            painter.close()  # type: ignore[attr-defined]
        return lama, 3, lama_clean, "lama"

    def replace(self, image: Image.Image, boxes: list[PixelBox], prompt: str,
                still_there: Callable[[Image.Image], list[PixelBox]] | None = None,
                new_there: Callable[[Image.Image, np.ndarray], bool] | None = None,
                max_change: float = MAX_CHANGE) -> EditResult:
        """Paint `prompt` over the objects. `clean` is False if the old object is still detected there; if `new_there`
        says the new thing isn't there, it paints once more with another seed and keeps a result that has it."""
        if self._paint_factory is None:
            raise RuntimeError("replacing objects needs the image model, which is not set up on this computer")
        with self._lock:
            t = time.perf_counter()
            raw = self.segment(image, boxes)
            step = max(2, round(GROW * max(image.size)))
            mask = grow(raw, step)
            if mask.mean() > max_change:
                raise EditRefused(f"it covers about {round(100 * float(raw.mean()))}% of the photo")
            if REPLACE_MASK == "box":  # room for the new thing's own shape, unless that would be too big a change
                boxed = _with_boxes(raw, boxes, step)
                if boxed.mean() <= max_change:
                    mask = boxed
            painter = self._paint_factory()
            seed = getattr(painter, "seed", None)
            try:
                def paint() -> Image.Image:
                    return fill_region(image, mask, lambda im, m: painter.paint(im, m, prompt),  # type: ignore[attr-defined]
                                       max_side=painter.max_side, context=REPLACE_CONTEXT)  # type: ignore[attr-defined]

                out, attempts = paint(), 1
                found = new_there(out, mask) if new_there is not None else None
                if found is False and hasattr(painter, "seed"):
                    painter.seed += 1  # type: ignore[attr-defined]
                    again, attempts = paint(), 2
                    if new_there(again, mask):  # type: ignore[misc]
                        out, found = again, True
            finally:
                if seed is not None:
                    painter.seed = seed  # type: ignore[attr-defined]  # a model kept warm starts the next edit the same
                painter.close()  # type: ignore[attr-defined]
            clean = None
            if still_there is not None:
                clean = not [b for b in still_there(out) if any(_box_iou(b, x) >= 0.3 for x in boxes)]
            return EditResult(out, mask, len(boxes), time.perf_counter() - t, attempts, clean, found)

    def models(self, replacing: bool = False) -> list[str]:
        names = [f"{SAM_ID}@{SAM_REVISION[:12]}"]
        if replacing:
            names.append(getattr(self._paint_factory, "model_name", "image model"))
        else:
            names.append(f"{LAMA_REPO}/{LAMA_FILE}@{LAMA_REVISION[:12]}")
        return names


class EditStore:
    """Edited photos, made-up images and videos live under <data>/edits as <id>.jpg|mp4 + <id>.json.
    Ids are random, never paths; the file name comes from the metadata this app wrote."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def save(self, image: Image.Image, meta: dict) -> dict:
        eid = secrets.token_hex(12)
        image.convert("RGB").save(self.root / f"{eid}.jpg", "JPEG", quality=JPEG_QUALITY)
        meta = {**meta, "id": eid, "file": f"{eid}.jpg", "width": image.width, "height": image.height,
                "created": time.time()}
        (self.root / f"{eid}.json").write_text(json.dumps(meta))
        return meta

    def new_path(self, suffix: str) -> Path:
        """Where to write a new file that adopt() will register (a video, written by FFmpeg)."""
        return self.root / f"{secrets.token_hex(12)}{suffix}"

    def adopt(self, path: Path, meta: dict) -> dict:
        eid = path.stem
        meta = {"width": None, "height": None, **meta, "id": eid, "file": path.name, "created": time.time()}
        (self.root / f"{eid}.json").write_text(json.dumps(meta))
        return meta

    def get(self, eid: str) -> dict | None:
        if not (len(eid) == 24 and all(c in "0123456789abcdef" for c in eid)):
            return None
        p = self.root / f"{eid}.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def path(self, eid: str) -> Path | None:
        meta = self.get(eid)
        if meta is None:
            return None
        return self.root / meta.get("file", f"{eid}.jpg")


def models_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    files = [(SAM_ID, f, SAM_REVISION) for f in ("config.json", "preprocessor_config.json", "model.safetensors")]
    files.append((LAMA_REPO, LAMA_FILE, LAMA_REVISION))
    paths = [try_to_load_from_cache(r, f, revision=v) for r, f, v in files]
    return all(isinstance(p, str) and Path(p).exists() for p in paths)


def download() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(SAM_ID, revision=SAM_REVISION,
                      allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
    hf_hub_download(LAMA_REPO, LAMA_FILE, revision=LAMA_REVISION)


def default_editor(device: str, painter_factory=None, keep_warm: float = 0) -> Editor:
    """keep_warm > 0: keep SAM and LaMa loaded that many seconds between edits (wrap painter_factory in KeepWarm too)."""
    if keep_warm <= 0:
        return Editor(lambda: SamSegmenter(device), lambda: LamaFiller(device), painter_factory)
    return Editor(KeepWarm(lambda: SamSegmenter(device), keep_warm), KeepWarm(lambda: LamaFiller(device), keep_warm),
                  painter_factory)


def main() -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Download and check the models used to edit photos in the Ask tab.")
    ap.add_argument("--download", action="store_true", help="download them (needs MEDIAINDEX_ALLOW_DOWNLOAD=1)")
    a = ap.parse_args()
    if a.download:
        if os.environ.get("MEDIAINDEX_ALLOW_DOWNLOAD") != "1":
            raise SystemExit("downloads are off by default; run with MEDIAINDEX_ALLOW_DOWNLOAD=1")
        download()
    if not models_cached():
        raise SystemExit(f"the edit models are not in the local cache; run: {DOWNLOAD_HINT}")
    im = Image.new("RGB", (64, 64), "white")
    r = default_editor("cpu").remove(im, [(16.0, 16.0, 48.0, 48.0)])
    print(f"ok: {SAM_ID} @ {SAM_REVISION[:12]} and {LAMA_REPO}/{LAMA_FILE} load and run ({r.seconds:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
