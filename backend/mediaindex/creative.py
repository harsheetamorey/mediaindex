"""Image model for the Ask tab: paint a new object into a photo ("replace the dog with a cat"), or make a new image.

One model does both, so only one is downloaded: Stable Diffusion 1.5 inpainting (CreativeML OpenRAIL-M licence, which
forbids some uses; see docs/create.md), in half precision, with the LCM-LoRA adapter (openrail++) so it needs about
6 steps instead of 50. A new image is the same model inpainting a blank canvas.

Optional: needs `uv sync --extra create` and a one-time download. Loads on first use and is freed afterwards.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

SD_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SD_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
SD_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"]
LCM_ID = "latent-consistency/lcm-lora-sdv1-5"
LCM_REVISION = "cf2fced511dbe7e26c8d1d397e728fbab875db4b"
LCM_FILE = "pytorch_lora_weights.safetensors"
MODEL_NAME = f"{SD_ID}@{SD_REVISION[:12]} + {LCM_ID}@{LCM_REVISION[:12]}"
DOWNLOAD_HINT = "uv sync --extra create && MEDIAINDEX_ALLOW_DOWNLOAD=1 uv run python -m mediaindex.creative --download"

STEPS = 6
GUIDANCE = 1.5  # LCM works with little or no guidance; a little keeps the picture closer to the words
NATIVE = 512  # the model's training size
STYLE = "a photo of {}, realistic, natural light, detailed, high quality"
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

    def __init__(self, device: str = "cpu", seed: int = 0):
        import torch
        from diffusers import LCMScheduler, StableDiffusionInpaintPipeline

        self._torch = torch
        self.device = device
        self.seed = seed
        dtype = torch.float16 if device != "cpu" else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_ID, revision=SD_REVISION, variant="fp16", torch_dtype=dtype, safety_checker=None,
            requires_safety_checker=False, feature_extractor=None)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(LCM_ID, weight_name=LCM_FILE, revision=LCM_REVISION)
        pipe.fuse_lora()
        pipe.vae.to(torch.float32)  # the fp16 VAE can overflow to black images on MPS
        self.pipe = pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def _run(self, image: Image.Image, mask: np.ndarray, prompt: str, size: tuple[int, int]) -> Image.Image:
        torch = self._torch
        w, h = size
        g = torch.Generator("cpu").manual_seed(self.seed)
        with torch.inference_mode():
            out = self.pipe(prompt=STYLE.format(prompt), negative_prompt=NEGATIVE,
                            image=image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS),
                            mask_image=Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST),
                            height=h, width=w, num_inference_steps=STEPS, guidance_scale=GUIDANCE, strength=1.0,
                            generator=g).images[0]
        return out

    def paint(self, image: Image.Image, mask: np.ndarray, prompt: str) -> Image.Image:
        """Paint `prompt` where mask is True; returns an image the size of `image` (the caller keeps only the mask)."""
        out = self._run(image, mask, prompt, _size(*image.size))
        return out.resize(image.size, Image.Resampling.LANCZOS)

    def generate(self, prompt: str, width: int = NATIVE, height: int = NATIVE) -> Image.Image:
        blank = Image.new("RGB", (width, height), (127, 127, 127))
        return self._run(blank, np.ones((height, width), dtype=bool), prompt, (width, height))

    def close(self) -> None:
        self.pipe = None
        if self.device == "mps":
            self._torch.mps.empty_cache()


def available() -> tuple[bool, str]:
    """Whether the optional package and the model files are present (filesystem only)."""
    try:
        import diffusers  # noqa: F401
        import peft  # noqa: F401
    except ImportError:
        return False, "the image model isn't installed"
    from huggingface_hub import try_to_load_from_cache

    files = [(SD_ID, f, SD_REVISION) for f in SD_FILES if "*" not in f]
    files.append((LCM_ID, LCM_FILE, LCM_REVISION))
    for r, f, v in files:
        p = try_to_load_from_cache(r, f, revision=v)
        if not (isinstance(p, str) and Path(p).exists()):
            return False, "the image model isn't downloaded yet"
    return True, ""


def download() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(SD_ID, revision=SD_REVISION, allow_patterns=SD_FILES)
    hf_hub_download(LCM_ID, LCM_FILE, revision=LCM_REVISION)


def painter_factory(device: str):
    def make() -> Painter:
        ok, why = available()
        if not ok:
            raise RuntimeError(f"{why}. Run this once in the MediaIndex folder: {DOWNLOAD_HINT}")
        return Painter(device)

    make.model_name = MODEL_NAME  # type: ignore[attr-defined]
    return make


def main() -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Download and check the image model used by the Ask tab.")
    ap.add_argument("--download", action="store_true", help="download it (needs MEDIAINDEX_ALLOW_DOWNLOAD=1)")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    if a.download:
        if os.environ.get("MEDIAINDEX_ALLOW_DOWNLOAD") != "1":
            raise SystemExit("downloads are off by default; run with MEDIAINDEX_ALLOW_DOWNLOAD=1")
        download()
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
