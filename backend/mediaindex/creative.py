"""Image model for the Ask tab: paint a new object into a photo ("replace the dog with a cat"), or make a new image.

Stable Diffusion 1.5 (CreativeML OpenRAIL-M licence, which forbids some uses; see docs/edit.md), in half precision,
with the LCM-LoRA adapter (openrail++) so it needs 6 steps instead of 50. The inpainting UNet both replaces objects and
makes new images (painting a blank canvas). The original text-to-image UNet ("generate" mode, optional extra 1.7 GB) was
measured too: it finds the asked-for thing as often and is twice as fast, but with LCM its pictures are harsh and
oversaturated whatever the prompt, so it isn't the default (docs/edit.md).

Optional: needs `uv sync --extra create` and a one-time download. Loads on first use; the app keeps it loaded for a few minutes (edit.KeepWarm).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import setup_step

SD_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SD_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
SD_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"]
LCM_ID = "latent-consistency/lcm-lora-sdv1-5"
LCM_REVISION = "cf2fced511dbe7e26c8d1d397e728fbab875db4b"
LCM_FILE = "pytorch_lora_weights.safetensors"
BASE_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # the text-to-image UNet; text encoder and VAE are shared
BASE_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
BASE_FILES = ["unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors"]
MODEL_NAME = f"{SD_ID}@{SD_REVISION[:12]} + {LCM_ID}@{LCM_REVISION[:12]}"
BASE_MODEL_NAME = f"{BASE_ID}@{BASE_REVISION[:12]} + {LCM_ID}@{LCM_REVISION[:12]}"
DOWNLOAD_HINT = "uv sync --extra create && MEDIAINDEX_ALLOW_DOWNLOAD=1 uv run python -m mediaindex.creative --download"

STEPS = 6
GUIDANCE = 1.5  # LCM works with little or no guidance; a little keeps the picture closer to the words
PAINT_GUIDANCE = 2.5  # replacing: more, or the model often paints background instead (55% -> 65% replaced; 4.0: 50%)
NATIVE = 512  # the model's training size
STYLE = "a photo of {}, realistic, natural light, detailed, high quality"
EMPTY = "an empty {}, nothing there, realistic photo"  # removing: what fills the hole ("road" where a car was)
EMPTY_NEGATIVE = "car, van, vehicle, truck, bus, person, people, animal, object, blurry, smudge, blocky"
NEGATIVE = "blurry, deformed, distorted, low quality, text, watermark, cartoon, drawing, frame, border"


def _size(w: int, h: int, side: int = NATIVE) -> tuple[int, int]:
    """Scale so the shorter side is `side` (longer at most 1.5x that), both multiples of 8."""
    s = side / min(w, h)
    nw, nh = w * s, h * s
    cap = 1.5 * side / max(nw, nh)
    if cap < 1:
        nw, nh = nw * cap, nh * cap
    return max(8, int(round(nw / 8)) * 8), max(8, int(round(nh / 8)) * 8)


class Painter:
    model_name = MODEL_NAME
    max_side = 768  # the crop around the object is scaled to this before painting

    def __init__(self, device: str = "cpu", seed: int = 0, mode: str = "paint"):
        """mode "paint": the inpainting model (replace objects); "generate": the text-to-image UNet for new images."""
        import torch
        from diffusers import LCMScheduler, StableDiffusionInpaintPipeline, StableDiffusionPipeline, UNet2DConditionModel

        self._torch = torch
        self.device = device
        self.seed = seed
        self.mode = mode
        dtype = torch.float16 if device != "cpu" else torch.float32
        common = dict(torch_dtype=dtype, safety_checker=None, requires_safety_checker=False, feature_extractor=None)
        if mode == "generate":
            unet = UNet2DConditionModel.from_pretrained(BASE_ID, subfolder="unet", revision=BASE_REVISION,
                                                        variant="fp16", torch_dtype=dtype)
            pipe = StableDiffusionPipeline.from_pretrained(SD_ID, revision=SD_REVISION, variant="fp16", unet=unet,
                                                           **common)
        else:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_ID, revision=SD_REVISION, variant="fp16", **common)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)  # before fusing: fusing in half precision on the CPU took 14 s, on the GPU 2 s
        pipe.load_lora_weights(LCM_ID, weight_name=LCM_FILE, revision=LCM_REVISION)
        pipe.fuse_lora()
        self.pipe = pipe
        self.pipe.enable_attention_slicing()  # 8 GB Macs: ~12 s per image instead of minutes of swapping
        self.pipe.set_progress_bar_config(disable=True)

    def _run(self, image: Image.Image, mask: np.ndarray, prompt: str, size: tuple[int, int],
             guidance: float | None = None) -> Image.Image:
        torch = self._torch
        w, h = size
        g = torch.Generator("cpu").manual_seed(self.seed)
        with torch.inference_mode():
            out = self.pipe(prompt=STYLE.format(prompt), negative_prompt=NEGATIVE,
                            image=image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS),
                            mask_image=Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST),
                            height=h, width=w, num_inference_steps=STEPS, guidance_scale=guidance or GUIDANCE, strength=1.0,
                            generator=g).images[0]
        return out

    def paint(self, image: Image.Image, mask: np.ndarray, prompt: str) -> Image.Image:
        """Paint `prompt` where mask is True; returns an image the size of `image` (the caller keeps only the mask)."""
        out = self._run(image, mask, prompt, _size(*image.size), PAINT_GUIDANCE)
        return out.resize(image.size, Image.Resampling.LANCZOS)

    def clear(self, image: Image.Image, mask: np.ndarray, strength: float = 1.0, place: str = "background") -> Image.Image:
        """Paint empty `place` where mask is True (removing a big object). strength < 1 repaints what is there
        lightly (it keeps its layout) instead of painting from scratch. Saying the place matters: "an empty road"
        filled a removed car seamlessly, "empty background" painted a new car."""
        torch = self._torch
        w, h = _size(*image.size)
        g = torch.Generator("cpu").manual_seed(self.seed)
        with torch.inference_mode():
            out = self.pipe(prompt=EMPTY.format(place), negative_prompt=EMPTY_NEGATIVE,
                            image=image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS),
                            mask_image=Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST),
                            height=h, width=w, num_inference_steps=max(STEPS, round(STEPS / strength)),
                            guidance_scale=PAINT_GUIDANCE if strength >= 1 else GUIDANCE, strength=strength,
                            generator=g).images[0]
        return out.resize(image.size, Image.Resampling.LANCZOS)

    def generate(self, prompt: str, width: int = NATIVE, height: int = NATIVE) -> Image.Image:
        if self.mode != "generate":  # the inpainting model on a blank canvas
            blank = Image.new("RGB", (width, height), (127, 127, 127))
            return self._run(blank, np.ones((height, width), dtype=bool), prompt, (width, height))
        torch = self._torch
        g = torch.Generator("cpu").manual_seed(self.seed)
        with torch.inference_mode():
            return self.pipe(prompt=STYLE.format(prompt), negative_prompt=NEGATIVE, height=height, width=width,
                             num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=g).images[0]

    def close(self) -> None:
        self.pipe = None
        if self.device == "mps":
            self._torch.mps.empty_cache()


def available(mode: str = "paint") -> tuple[bool, str]:
    """Whether the optional package and the model files are present (filesystem only)."""
    try:
        import diffusers  # noqa: F401
        import peft  # noqa: F401
    except ImportError:
        return False, "the image model isn't installed"
    from huggingface_hub import try_to_load_from_cache

    files = [(SD_ID, f, SD_REVISION) for f in SD_FILES if "*" not in f]
    files.append((LCM_ID, LCM_FILE, LCM_REVISION))
    if mode == "generate":
        files += [(BASE_ID, f, BASE_REVISION) for f in BASE_FILES]
    for r, f, v in files:
        p = try_to_load_from_cache(r, f, revision=v)
        if not (isinstance(p, str) and Path(p).exists()):
            return False, "the image model isn't downloaded yet"
    return True, ""


def download(base: bool = False) -> None:
    """base: also the text-to-image UNet (only for comparing in evaluation/bench_create.py)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(SD_ID, revision=SD_REVISION, allow_patterns=SD_FILES)
    if base:
        snapshot_download(BASE_ID, revision=BASE_REVISION, allow_patterns=BASE_FILES)
    hf_hub_download(LCM_ID, LCM_FILE, revision=LCM_REVISION)


def painter_factory(device: str):
    """make(mode): "paint" (the inpainting model, used for everything) or "generate" (text-to-image UNet)."""
    def make(mode: str = "paint") -> Painter:
        ok, why = available(mode)
        if not ok:
            raise RuntimeError(f"{why}. {setup_step(DOWNLOAD_HINT)}")
        return Painter(device, mode=mode)

    make.model_name = MODEL_NAME  # type: ignore[attr-defined]
    make.generate_model_name = BASE_MODEL_NAME  # type: ignore[attr-defined]
    return make


def main() -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Download and check the image model used by the Ask tab.")
    ap.add_argument("--download", action="store_true", help="download it (needs MEDIAINDEX_ALLOW_DOWNLOAD=1)")
    ap.add_argument("--base", action="store_true", help="also the text-to-image UNet (for the benchmark only)")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    if a.download:
        if os.environ.get("MEDIAINDEX_ALLOW_DOWNLOAD") != "1":
            raise SystemExit("downloads are off by default; run with MEDIAINDEX_ALLOW_DOWNLOAD=1")
        download(a.base)
    ok, why = available()
    if not ok:
        raise SystemExit(f"{why}; run: {DOWNLOAD_HINT}")
    t = time.perf_counter()
    p = Painter(a.device)
    im = p.generate("a red apple on a wooden table")
    print(f"ok: {MODEL_NAME} loads and runs ({im.size[0]}x{im.size[1]}, {time.perf_counter() - t:.1f} s incl. loading)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
