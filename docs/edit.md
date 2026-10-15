# Edit, create and make videos in Ask

In the **Ask** tab you can change a photo or make something new by asking in plain words. Everything runs on this computer.

> **You:** Show me photos with cars. → *[12 photos]*
> **You:** Remove the car from photo 2. → *[a new copy of photo 2 without the car]*
> **You:** Now remove the person too. → *[keeps editing that copy]*
> **You:** Replace the dog with a cat in the first one.
> **You:** Make an image of a red bicycle on a beach. → *[a made-up image, labelled as such]*
> **You:** Turn those into a reel. → *[a 9:16 video of exactly those photos]*

- **Your originals never change.** Every result is a new file in MediaIndex's own data folder, shown with **Show original**
  and **Save a copy…**. Saving asks for a folder outside your libraries and never overwrites a file.
- **Only the object's area changes.** Every pixel outside the (slightly grown) outline is copied from the original. The tests
  and the benchmark check this.
- **It says when it isn't sure.** After removing something it checks the result again. If the object is still found there,
  it fills a wider area once more, and says so if it still isn't clean. It declines objects that cover more than about a
  third of the photo, and warns when a big area was filled.
- **Edits must be asked for in words** ("remove", "erase", "get rid of", "replace … with …"). The chat model never decides
  to edit. "Delete photo 3" is answered with "MediaIndex never deletes or changes your photos".

## How it works

| Step | Model | Licence | Size |
|---|---|---|---|
| Find the object you named | **RT-DETR v2** (the detector Ask already uses; 80 common object types) | Apache-2.0 | 170 MB |
| Outline it exactly | **SAM ViT-B** (`facebook/sam-vit-base`, pinned revision), prompted with the detector's box | Apache-2.0 | 375 MB |
| Remove it: fill from the surroundings | **LaMa** (`big-lama`, TorchScript from `fashn-ai/LaMa`, pinned revision and SHA-256) | Apache-2.0 | 206 MB |
| Replace it, or make a new image | **Stable Diffusion 1.5 inpainting** + **LCM-LoRA** (6 steps instead of 50), fp16 | CreativeML OpenRAIL-M / openrail++ | 2.1 GB |
| Make a video of photos | No model: Pillow draws the frames, FFmpeg encodes H.264 | — | — |

- **Which object:** "the car" removes the biggest car the detector is sure of; "the cars" or "all the cars" removes every
  one. If there are more, the answer says how many and how to remove them all.
- **Outline choice:** SAM offers three outlines (a part, the object, a group). Picking the one whose extent best matches the
  detector's box, rather than SAM's own top score, fixed cases where SAM outlined only a child's face.
- **Fill:** LaMa works on a crop around the object, with as much surrounding context as the object's size, at up to
  1024 px. Lower resolutions (512, 384) were tried on the hard cases and were no better.
- **Videos:** each photo is shown for 3 s with a slow zoom and a 0.6 s crossfade. Photos that don't fit the shape are shown
  whole over a blurred copy instead of being cropped. Shapes: 16:9 (default), 9:16 ("reel", "short", "story"), square.
  Measured on an M1: 8 of your photos → a 20 s 1080p video in **11 s** (a 9:16 reel: 12 s), every frame present. Frames
  are drawn by Pillow on 4 threads (~11 ms each); torch on the GPU took 25 s for the same video.
- **Made-up images are labelled** in the answer, in the app ("Made-up image") and in Finder tags when saved ("Made up").

## Setup

```bash
MEDIAINDEX_ALLOW_DOWNLOAD=1 uv run python -m mediaindex.edit --download        # remove: SAM + LaMa (~580 MB)
uv sync --extra create                                                          # replace / make images: diffusers, peft
MEDIAINDEX_ALLOW_DOWNLOAD=1 uv run python -m mediaindex.creative --download    # Stable Diffusion inpainting + LCM (~2.1 GB)
```

The server never downloads. If a model is missing, the answer gives the command to run. Making videos needs only FFmpeg.

## Measured: removing objects (REAL MODEL, M1 8 GB)

`evaluation/bench_edit.py`, 100 COCO val2017 photos (the same 300-photo sample as the counting check, human-drawn object
outlines). One task per photo: the largest object the detector is sure of that matches a human-labelled object covering
0.5–25% of the photo. Run exactly as in the app. Results in `evaluation/bench/edit_remove.json`.

| Measure | Result |
|---|---|
| Outline vs the human outline (IoU) | mean **0.84**, median **0.88** |
| Object gone (the detector, at a stricter 0.3 threshold, no longer finds it there) | **92 / 100** |
| … when the app didn't warn "may not be clean" | **92 / 95** (5 edits were flagged in the answer) |
| Other objects outside the edit still detected afterwards | **271 / 271** |
| Pixels outside the edit unchanged | **100 / 100** |
| Time per edit (models loaded each time, plus the re-check) | median about 14 s in the benchmark (it outlines twice to score the outline) |

**Looked at by eye** (10 random "gone" results): 7 look clean (a cat in front of a TV, a person by a train, zebras, an oven,
a skier…). 3 have visible leftovers: a smudge where a large person was, a snowboarder's shadow still on the snow, and a dark
patch where a toilet was. So the detector-based 92% is an upper bound on "looks clean"; on everyday photos, expect most
small and medium objects to come out cleanly and large ones to leave traces.

Changes made because of this benchmark:
- choosing SAM's outline by the detector's box (removed 83% → 90% on a 30-photo check);
- re-checking and filling a wider area once when the object is still found (90% → 92% on 100);
- declining edits that would change more than 35% of a photo: a close-up arm had made a retry repaint 92% of the photo.

## Measured: replacing objects (REAL MODEL, M1 8 GB)

`evaluation/bench_create.py replace`: the same kind of tasks as removing (one sure, human-confirmed object per COCO photo),
replaced with a fixed other kind of thing (person → dog, car → truck, zebra → horse, cake → pizza…). Success = the
detector (at 0.3) no longer finds the old thing there **and** finds the new thing mostly inside the painted area.

| Version | 40 photos (seed 0) | 30 new photos (seed 1) |
|---|---|---|
| First version: exact outline, 3× surroundings, guidance 1.5 | 16 / 40 (40%) | — |
| Less surroundings (0.5×), so the object is bigger on the model's canvas | 20 / 40 | — |
| … paint the object's box, not its outline (box refused when > 35% of the photo) | 19 / 40 | 11 / 30 |
| … guidance 2.5 instead of 1.5 (4.0 was worse: 50%) | — | 13 / 30 |
| **Now:** box (outline if the box is too big), guidance 2.5, one retry if the new thing isn't found | **28 / 40 (70%)** | **19 / 30 (63%)** |
| … same, without the retry | 25 / 40 | — |

Pixels outside the painted area were unchanged in every edit. Median time **35 s** per replacement with the model already
loaded; 13 of 40 edits needed the retry. In the app, which loads the models for each edit, a replacement took 89 s end
to end (a car → a believable white van in a car park; "now remove the truck" then edited that copy in 14 s).

**Loading faster (2026-10-15):** most of the loading was merging the LCM adapter into the model in half precision on the
CPU (14 s). It is now merged after the model is on the GPU (2 s): loading the image model went from 31 s to 13 s on an
M1, with the same pictures (mean pixel difference 0.07 of 255). The app also keeps the edit models loaded for 5 minutes
after an edit (`KEEP_WARM_S`), but only while at least 1.5 GB of memory stays available. On an 8 GB Mac that isn't the
case with Stable Diffusion loaded (0.7 GB left, measured), so there it is still freed after every edit; machines with
more memory skip loading on the next edit.

**Caveat:** the retry uses the same detector that scores the benchmark, so its +3 is an upper bound. **Looked at by eye**
(12 random successes): all 12 show the asked-for thing, 11 believably (a fluffy dog where a baby was, a bus, a pick-up truck,
pizza on the cake plate); one, a dog standing in a tennis player's pose, is clearly a dog but odd. In failures the model
mostly paints background (the object disappears), and the answer then says the check didn't find the new thing.

## Measured: making new images (REAL MODEL, M1 8 GB)

`evaluation/bench_create.py generate`: "a <thing>" for 24 of the detector's object types plus 12 scenes ("a dog on a
beach at sunset", "a bus in the rain", "a person with an umbrella on a rainy street"…), 2 seeds each, 72 images. Found =
the detector finds the asked-for thing (score ≥ 0.5). The detector is only a proxy judge, so the images were also looked at.

| Model (both with LCM-LoRA, 6 steps) | Found | Scenes found | Time per 512² image | Looks |
|---|---|---|---|---|
| **SD 1.5 inpainting on a blank canvas** (the app's choice) | **71 / 72** | 24 / 24 | about 15–25 s | natural colours and light |
| SD 1.5 text-to-image UNet (extra 1.7 GB) | 72 / 72 | 24 / 24 | about 9 s | harsh contrast, oversaturated |

The text-to-image model's look stayed harsh with guidance 1.0 and with plainer prompts, so the inpainting model is used for
everything (also no extra download). Guidance 1.0 made the inpainting model washed out and blurry, so it stays at 1.5. The
one miss was an elephant cut off by the frame. Attention slicing is required on 8 GB: without it one image took ~9 minutes.
Results in `evaluation/bench/create_generate-paint.json` and `create_generate-base.json`.

## Limits

- **Only the 80 object types the detector knows** can be removed or replaced (people, cars, dogs, cats, birds, bikes,
  boats, cups…), not "the tree" or "the sign".
- **Shadows and reflections stay** unless the detector's outline covers them.
- **Crowded scenes fill badly.** Removing one car from a full car park (11% of the photo, no warning) left a dark,
  blocky patch: there is no clean background nearby to copy from.
- **Big removals (over 12% of the photo), since 2026-10-15:** LaMa fills the object's whole box at 512 px (fewer blocks
  than at 1024), then the image model paints "an empty road" (vehicles) or "an empty background" there, and if that
  isn't accepted, lightly repaints LaMa's fill (strength 0.6). A try is kept only if the detector finds nothing new in
  the filled area: the image model often paints a new car where a car was. Checked by eye on two photos only: a
  sports car on a night road is now filled with road and kerb (before: a dark smear); a VW van in a busy junction
  still falls back to LaMa (every image-model try added a car), with less blocking but a smudge and its shadow left.
  Not yet measured on the 100-photo benchmark. "Make it seamless" or "try again" after a removal tries another seed.
- **Large objects leave smudges.** Anything over about 12% of the photo gets a warning; over about 35% is declined.
- **Replacing and new images are made up** by a small image model (Stable Diffusion 1.5): faces, hands and text are often
  wrong. Its licence (CreativeML OpenRAIL-M) forbids some uses; read it before publishing results.
- **Videos are slideshows of your photos,** not generated video. Generating video with a diffusion model doesn't fit in
  8 GB of memory at a usable speed, so MediaIndex doesn't pretend to.
