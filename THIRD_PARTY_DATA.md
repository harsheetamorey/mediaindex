# Third-party data

MediaIndex does not bundle any third-party media in this repository. Optional demo data is downloaded on request
to `data/demo/`, which is git-ignored.

## Optional demo image pack: KoalaAI/StockImages-CC0
- Source: https://huggingface.co/datasets/KoalaAI/StockImages-CC0, pinned revision `206f3575579f1187548c6f47042ae9174c0a51fc`.
- Download: `uv run --extra demo python scripts/download_demo.py` gets 500 images and transfers about 115 MB using HTTP range reads of 5 parquet row groups.
- Selection: deterministic (fixed seed, pinned revision). The committed manifest `manifests/stockimages-cc0-500.json` lists
  row IDs and SHA-256 hashes, and the script verifies every run against it.
- License: the **publisher declares CC0-1.0** for the dataset as a whole. The dataset card says the images were "compiled from various
  sources" but does **not** include per-image provenance (original photographer, source URL or license evidence).
- MediaIndex records, per image, the dataset, revision, row number, SHA-256, publisher-declared license and dataset URL in
  `mediaindex-provenance.json` next to the images. This is shown and exported as the asset's source record.
- **No independent rights audit has been performed.** A publisher's declaration is not a rights clearance. Verify suitability
  before reusing any image in published work.
- The dataset's `tags` field is kept as inspection metadata only. It is never used in embeddings or as ground truth.
- Known data quality issue: some rows in this dataset hold HTML error pages instead of image bytes. The downloader excludes
  undecodable rows and records them in the manifest. The pinned 500-image selection has none.
