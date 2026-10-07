# Image + text refinement: observed behaviour (Phase 8)

Run on 2026-10-07 against the 500-image demo pack (MPS bf16, profile revision `914f7f8…`) with
`uv run python scripts/refinement_check.py --library <demo library id>`.

Every row compares the top 5 for **image only** (stored vector of the reference), **text only**, and **native image+text**.
The image+text embedding comes from one forward pass:
`SentenceTransformer.encode({"text": "task: search result | query: <text>", "image": <PIL image>})`. It is not an average of two vectors.

Publisher tags were used **only to eyeball** results. They are not ground truth, and nothing below is an accuracy metric.

| Reference (what it shows) | Refinement | Observation | Verdict |
|---|---|---|---|
| stock-00102 (purple viola macro) | "yellow flowers" | Ranks 2–3 move to the two yellow-tagged flower images. The near-duplicate purple viola stays at #1. | Text helped partly. Colour shifted, but the reference still dominates the top. |
| stock-00150 (Paris skyline, night) | "at night" | Daytime streets and architecture give way to night city and night street scenes (4/5 night-tagged versus 0/5 for image only). | Text helped. |
| stock-00171 (snowy mountains, lake) | "at sunset" | Snowy forests give way to sunrise/sunset hills and mountain horizons (0 overlap with image only). The scenes stay landscapes, unlike text-only, which returned sunsets at sea. | Text helped, and the combination kept the landscape context. |
| stock-02096 (rocky coast, colour) | "black and white photo" | Identical top 5 to image only (5/5 overlap). No monochrome image appears. | **Text ignored.** |
| stock-03482 (sports car) | "a car" | Image only fails (night and road scenes). Image+text brings two car images into the top 5 (#3 and #5), but #1 and #2 are unrelated images. | Partial improvement. Still unreliable. |
| stock-00102 (purple viola) | "without any flowers" | Results are still mostly flowers. | **Negation not respected.** |

## Takeaways
- Refinement text does change the query. Setting and lighting cues ("at night", "at sunset") had the clearest effect.
- When the reference has strong near-duplicates, the reference tends to dominate and the text can be ignored ("black and white photo").
- Negation and logical constraints are **not** supported. Words like "without" just add the concept to the query.
- An API success alone does not show that a refinement worked. Check results visually. These six cases are anecdotes.
  Phase 17 adds a labelled evaluation.
