"""Look-alike photos: burst shots, re-saved, cropped or edited copies, grouped for review. Nothing is ever deleted.

Groups come from the stored EmbeddingGemma image vectors (cosine similarity, not confidence). Thresholds were set with
the real model (evaluation/bench_lookalikes.py, docs/progress.md): on the 500-photo stock library, copies made smaller,
cropped, brighter or tilted scored 0.87-0.99 against their original, and different photos above 0.92 were nearly all
shots of the same scene. Each group is a star around one photo, so a chain of similar scenes never snowballs into one
big group.
"""

from __future__ import annotations

import numpy as np

LEVELS = {"strict": 0.95, "normal": 0.92, "loose": 0.90}
BLOCK = 1024


def group_lookalikes(ids: list[str], matrix: np.ndarray, threshold: float) -> list[list[str]]:
    """Groups of 2+ ids whose vectors are within `threshold` of the group's first photo. Biggest groups first."""
    n = len(ids)
    if n < 2:
        return []
    neighbours: list[np.ndarray] = []
    for s in range(0, n, BLOCK):  # never the full n x n matrix at once
        sims = matrix[s:s + BLOCK] @ matrix.T
        for k, row in enumerate(sims):
            hit = np.flatnonzero(row >= threshold)
            neighbours.append(hit[hit != s + k])
    order = sorted(range(n), key=lambda i: (-len(neighbours[i]), ids[i]))  # best-connected photos lead a group
    taken = np.zeros(n, bool)
    groups = []
    for i in order:
        if taken[i] or len(neighbours[i]) == 0:
            continue
        members = [j for j in neighbours[i] if not taken[j]]
        if not members:
            continue
        taken[i] = True
        taken[members] = True
        groups.append([ids[i], *(ids[j] for j in sorted(members, key=lambda j: -float(matrix[j] @ matrix[i])))])
    return sorted(groups, key=lambda g: -len(g))


def sharpness(im) -> float:
    """How sharp a photo looks: variance of the Laplacian of its grey levels (pixels only, no model). Higher is
    sharper; blur from motion or focus lowers it. Comparable within a group of look-alikes, not across scenes."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def best_shot(items: list[tuple[str, float, int]]) -> str:
    """items: (id, sharpness, pixels). The sharpest; among shots within 10% of it, the biggest (an original beats its
    smaller re-saved copy). Measured on the stored thumbnails: a 1.5 px blur on the full photo was caught 40/40."""
    top = max(s for _, s, _ in items)
    close = [it for it in items if it[1] >= 0.9 * top]
    return max(close, key=lambda it: (it[2], it[1]))[0]
