# Ask: chat with your photo library

The **Ask** tab answers questions about your photos in plain words. Everything runs on this computer, and nothing is uploaded.

> **You:** How many dogs do I have?
> **MediaIndex:** Dogs appear in 3 of 512 photos (3 dogs counted). *[the 3 photos]*
> **You:** Show me a city street at night. → *[12 photos]*
> **You:** Which of those have cars? → 9 of those 12 photos have cars.
> **You:** Describe the first one. → "A nighttime photograph of a city street with numerous blurred lights…"

Click any photo in an answer to ask about that photo. Counted objects are **outlined** in the photos, so you can check every
answer. Doubting an answer ("why did you say it's a dog?", "where is it?", "that's not a cat") shows the evidence: the outlined
object and how sure the detector was. Ask doesn't guess.

## How it works: three local models, one job each

| Job | Model | Licence | When it runs |
|---|---|---|---|
| Find photos ("beach at sunset"), re-rank follow-ups | **EmbeddingGemma 2** (the same index as Search) | Gemma Terms | Each question |
| Count objects ("how many dogs?") | **RT-DETR v2** (`PekingU/rtdetr_v2_r50vd`, pinned revision), 80 common object types | Apache-2.0 | Once per photo, when you press **Count objects** |
| Understand oddly phrased questions, describe a photo, small talk | **Gemma 4 E2B** (`gemma4:e2b-it-qat`, 4-bit) through **Ollama** | Apache-2.0 | Each question that needs it (optional) |

- **Counting rule:** the same for all 80 types, chosen by measurement (see [below](#how-the-counting-rule-was-chosen)).
  - A photo counts when the detector scores the object at least **0.75**.
  - Photos where it only scores 0.5–0.75 are shown last, with a dashed border and a **maybe** tag, and aren't counted.
- **Your corrections win.** Say "photo 3 isn't a dog" or "1, 2 and 4 aren't umbrellas", and those photos are left out of
  that answer from then on. They are stored in the local database.
- **Descriptions get the detector's findings as a hint**, so they agree with the counts. Before this, Gemma 4 described
  a Yorkshire terrier the detector had counted as a dog as a "cat", and left out a small dog on a beach.
- **Numbers always come from the database, never from the chat model.** Counts, searches and follow-ups are answered with fixed
  sentences. If a free chat reply mentions a number that isn't in the facts the model was given, MediaIndex replaces it with a
  fixed summary.
- **Clear questions skip the chat model.** "How many…", "show me…", "which of those…" and "describe the second one" are routed
  by simple rules, so they answer instantly. Gemma 4 is only used for everything else.
- **Follow-ups:**
  - "Which of those have *cars*?" **filters** the previous photos by the object counts, when the subject is one of the 80 types.
  - Otherwise ("which ones are at sunset?"), it **re-orders** them by similarity and says that it can't be sure which ones match.
- **Without Ollama,** Ask still counts, searches and narrows down, and describing a photo lists the objects found instead.

Searching still uses EmbeddingGemma only. RT-DETR and Gemma 4 are helpers for Ask; they never change search results.

## Setup

1. **Object counts:** download the detector once, then open **Ask** and press **Count objects**.
   - `MEDIAINDEX_ALLOW_DOWNLOAD=1 uv run python -m mediaindex.detect --download` fetches RT-DETR (~170 MB, pinned
     revision) and checks that it runs. The server itself never downloads; if the detector is missing, **Count objects**
     fails with this command in the message.
   - The Mac app downloads it together with EmbeddingGemma on first launch.
   - Counting is about 0.3 s per photo; 512 photos took **145–151 s** on an M1.
   - New photos are added by pressing the button again.
2. **Chat model (optional):**
   - Install [Ollama](https://ollama.com/download). The Mac app version starts by itself at login.
   - Run `ollama pull gemma4:e2b-it-qat` (4.3 GB, Apache-2.0).
   - MediaIndex talks to it only at `http://127.0.0.1:11434`. Any non-local address is refused.
   - Settings:
     - `MEDIAINDEX_CHAT_MODEL` picks another model; set it to empty to turn chat off.
     - `MEDIAINDEX_OLLAMA_URL` must be a loopback address.
   - Don't use the default `gemma4:e2b` tag on an 8 GB Mac: it is 7.5 GB.

## How the models were chosen (M1, 8 GB; REAL MODEL checks)

These checks used 20 demo photos (dogs, cats, crowds, streets, birds), picked by EmbeddingGemma search. Counts were
checked by looking at the photos; these are **not** human evaluation labels.

| Model | Time per photo | Counting | Notes |
|---|---|---|---|
| **RT-DETR v2** (chosen) | **0.27 s** | Good on crowds (19 vs ~20 people; 12 vs ~10) and cars; birds 12/~14, 3/3, 3/3, 7/~8 | One small dog came out as "cow"; a lizard (not one of the 80 types) came out as "bird" |
| Florence-2 base | 2.0–2.3 s | Missed most people in crowds (6 vs ~20+, 4 vs ~10); messy labels ("man", "woman", "human face" for one crowd) | Good captions; 7/7 dogs and cats correct |
| Gemma 3 1B (text only) | 1.4 s per routing, 1.9 s per reply | n/a | Routed 9/10 test questions correctly; one reply garbled the numbers. Gated licence |
| **Gemma 4 E2B** 4-bit (chosen) | 1–10 s per description | Not used for counting | 10 of 11 photos described correctly (called a Yorkshire terrier a "calico cat"). **Sounds: 0 of 3 right** (crickets and footsteps came out as "dog barking"), so Ask doesn't describe sounds |

## How the counting rule was chosen (REAL MODEL, M1 8 GB)

`scripts/coco_check.py` ran the detector on **300 photos from COCO val2017**, sampled with seed 0.
- COCO val2017 is a public set where people drew a box around every object of these 80 types.
- The detector was trained on COCO *train*2017, so these photos are new to it.

Scores are at photo level, which is what Ask answers.
- **Precision:** of the photos reported to contain the thing, the share that really do.
- **Recall:** of the photos that contain it, the share that were found.

| Rule (all 80 types together) | Precision | Recall |
|---|---|---|
| score ≥ 0.5 | 0.865 | 0.874 |
| **score ≥ 0.75 (chosen)** | **0.962** | 0.714 |
| ≥ 0.75, or 0.5–0.75 when Gemma 4 says yes (whole-photo yes/no) | 0.923 | 0.795 |

- **Within the 0.5–0.75 band, only about 60% of photos were right** (140 of 235). That's why they're shown as "maybe"
  instead of counted.
- **Gemma 4 as a double-checker helped little.** Its "no" answers were right 36% of the time for boxes under 0.5% of the
  photo, and 73% for boxes over 5%. It isn't used for counting.
- **Per type, at 0.75,** precision was 0.75–1.00 for every type with at least 5 photos, except potted plants (0.60). Examples: people 0.98, cars 1.00,
  dogs 1.00, cats 1.00, umbrellas 1.00, bottles 0.79, potted plants 0.60. The full list is printed by
  `python scripts/coco_check.py report`.
- **Unusual photos are harder.** On the artistic demo photos, the detector still scored a petunia 0.94 and a sunflower 0.79
  as "umbrella". No score rule catches confident mistakes like these, which is why answers show their photos with the
  boxes, and why corrections are kept.

Earlier, before this measurement, I tried rules picked by eye from about 25 demo photos: separate rules for animals, and
Gemma 4 double-checking "other objects". They fixed the photos I looked at, but broke others: Gemma 4 rejected a real
Yorkshire terrier and small real cars. They were replaced by the measured rule above.

**End-to-end (real models, this Mac):**
- 512 photos counted in 151 s.
- **Dogs:** 3 photos, all real dogs; the 2 "maybe" photos have no dog.
- **Cats:** 4 photos, all real cats; the 1 "maybe" photo has none.
- **Umbrellas:** 8 photos, mostly wrong, from the confident mistakes above.
- **Timing:** count questions answer in under 0.1 s, searches in about 0.5 s (EmbeddingGemma's first load adds ~15 s), and
  describing a photo takes 7–15 s.
- `scripts/ui_check_ask.py` drives the Ask tab in Chrome.

## Limits
- **Only 80 common object types can be counted:** people, animals, vehicles and household things. Anything else ("sunsets",
  "lizards") gets the closest photos instead, and the answer says so.
- **Counts can be wrong.** On everyday photos (COCO), 96% of counted photos were right, and 29% of photos with the thing were
  missed. Unusual or artistic photos produce confident mistakes. Every answer shows its photos and boxes, and corrections are
  kept.
- **Photos only.** Sounds and videos aren't counted or described.
- **Gemma 4 can describe things that aren't there.** Treat descriptions as a quick look, not a record.
