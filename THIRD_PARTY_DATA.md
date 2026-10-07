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

## Optional demo sound pack: quinnlue/FSD50K-16k (CC0 clips only)
- Source: https://huggingface.co/datasets/quinnlue/FSD50K-16k, pinned revision `2a60d475f4e2f0db624a902881af2df79d656145`.
  This is a 16 kHz mono FLAC repack of FSD50K (Fonseca et al., 2020). (The build guide's Phase 11 link reads "F50K-16k". That repository does not
  exist, and the guide's source list gives FSD50K-16k.)
- Download: `uv run --extra demo python scripts/download_demo_audio.py` gets 60 clips, about 17 MB transferred (2 parquet row groups of the
  `validation` split).
- **Only clips whose own `license` field is CC0 1.0 are selected.** In the 128 rows read, 64 non-CC0 clips (CC BY, CC BY-NC, Sampling+) were skipped.
  The committed manifest `manifests/fsd50k-cc0.json` lists rows, Freesound IDs and SHA-256 hashes.
- Each clip's Freesound ID, URL, uploader, title and declared licence are kept in `mediaindex-provenance.json`.
- FSD50K as a curated collection is CC BY 4.0. Please cite: Fonseca, E., Favory, X., Pons, J., Font, F., Serra, X. *FSD50K: An Open Dataset
  of Human-Labeled Sound Events*, arXiv:2010.00475 (2020).
- Licence declarations come from Freesound uploaders through the dataset. **No independent rights audit has been performed.**
- FSD50K labels are kept as inspection metadata only and are never embedded.
