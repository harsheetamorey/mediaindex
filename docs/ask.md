# Ask: chat with your photo library

The **Ask** tab answers questions about your photos in plain words. Everything runs on this computer, and nothing is uploaded.

> **You:** How many dogs do I have?
> **MediaIndex:** Dogs appear in 5 of 512 photos (5 dogs counted). *[the 5 photos]*
> **You:** Show me a city street at night. → *[12 photos]*
> **You:** Which of those have cars? → 9 of those 12 photos have cars.
> **You:** Describe the first one. → "A nighttime photograph of a city street with numerous blurred lights…"

Click any photo in an answer to ask about that photo.

## How it works: three local models, one job each

| Job | Model | Licence | When it runs |
|---|---|---|---|
| Find photos ("beach at sunset"), re-rank follow-ups | **EmbeddingGemma 2** (the same index as Search) | Gemma Terms | Each question |
| Count objects ("how many dogs?") | **RT-DETR v2** (`PekingU/rtdetr_v2_r50vd`, pinned revision), 80 common object types | Apache-2.0 | Once per photo, when you press **Count objects** |
| Understand oddly phrased questions, describe a photo, small talk | **Gemma 4 E2B** (`gemma4:e2b-it-qat`, 4-bit) through **Ollama** | Apache-2.0 | Each question that needs it (optional) |

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

1. **Object counts:** open **Ask** and press **Count objects**.
   - RT-DETR (~170 MB) downloads once from Hugging Face.
   - Counting is about 0.3 s per photo; 512 photos took **145 s** on an M1.
   - New photos are added by pressing the button again.
   - In the Mac app, the detector has to be in the Hugging Face cache already, because the app runs offline once
     EmbeddingGemma is cached. Run Ask once from source first, or start the app before EmbeddingGemma is downloaded.
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

**End-to-end (real models, this Mac):**
- 512 photos counted in 145 s.
- Count questions answer in under 0.1 s and searches in about 0.5 s. EmbeddingGemma's first load adds ~15 s.
- Describing a photo takes about 7–15 s. The chat model's first load adds ~20 s.
- `scripts/ui_check_ask.py` drives the Ask tab in Chrome: a count, a search, a follow-up filter, "describe the first one", and
  clicking a photo.

## Limits
- **Only 80 common object types can be counted:** people, animals, vehicles and household things. Anything else ("sunsets",
  "lizards") gets the closest photos instead, and the answer says so.
- **Counts can be wrong.** Small, far-away or overlapping objects are missed, and look-alikes are confused. That's why every
  answer shows its photos.
- **Photos only.** Sounds and videos aren't counted or described.
- **Gemma 4 can describe things that aren't there.** Treat descriptions as a quick look, not a record.
