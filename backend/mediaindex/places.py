"""Offline place names for GPS positions: the nearest town from a bundled GeoNames list (cities15000, CC BY 4.0).

Nothing is looked up online. The list holds every place with 15,000+ people (scripts/build_places.py), so a beach or
a village gets a nearby town: within NEAR_KM it is named ("Calangute, Goa, India"); further out, only the nearest
town's region and country ("Goa, India"); beyond FAR_KM, no place at all. Among the towns within NEAR_KM, a big city
a little further away wins over a small one next to the photo (population / (km + 1)^2): the Eiffel Tower is in
"Paris", not in one of its suburbs.
"""

from __future__ import annotations

import gzip
import threading
import unicodedata
from pathlib import Path

import numpy as np

PLACES_FILE = Path(__file__).resolve().parent / "resources" / "places.tsv.gz"
NEAR_KM = 30.0
FAR_KM = 150.0
EARTH_KM = 6371.0

_lock = threading.Lock()
_table: tuple[list[tuple[str, str, str]], np.ndarray, np.ndarray] | None = None


def _load() -> tuple[list[tuple[str, str, str]], np.ndarray, np.ndarray]:
    global _table
    with _lock:
        if _table is None:
            names, coords, pops = [], [], []
            with gzip.open(PLACES_FILE, "rt", encoding="utf-8") as f:
                for line in f:
                    name, region, country, lat, lon, pop = line.rstrip("\n").split("\t")
                    names.append((name, region, country))
                    coords.append((float(lat), float(lon)))
                    pops.append(float(pop))
            _table = names, np.radians(np.asarray(coords, dtype=np.float64)), np.asarray(pops)
        return _table


def place_name(lat: float, lon: float) -> str | None:
    """ "Calangute, Goa, India" for a nearby town, "Goa, India" when the nearest one is further, else None."""
    names, rad, pops = _load()
    la, lo = np.radians(lat), np.radians(lon)
    h = np.sin((rad[:, 0] - la) / 2) ** 2 + np.cos(la) * np.cos(rad[:, 0]) * np.sin((rad[:, 1] - lo) / 2) ** 2
    km = 2 * EARTH_KM * np.arcsin(np.sqrt(np.minimum(1.0, h)))
    i = int(np.argmin(km))
    if km[i] > FAR_KM:
        return None
    if km[i] <= NEAR_KM:
        near = np.flatnonzero(km <= NEAR_KM)
        i = int(near[np.argmax(pops[near] / (km[near] + 1) ** 2)])
    name, region, country = names[i]
    parts = [name, region, country] if km[i] <= NEAR_KM else [region, country]
    return ", ".join(dict.fromkeys(p for p in parts if p))  # "Singapore, Singapore" -> "Singapore"


def fold(text: str) -> str:
    """Lower case without accents, for matching what people type: "São Paulo" -> "sao paulo"."""
    t = unicodedata.normalize("NFKD", text)
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()


def place_parts(place: str) -> list[str]:
    return [p.strip() for p in place.split(",") if p.strip()]


def place_levels(counts: dict[str, int]) -> list[dict]:
    """Every country, region and town in `counts` (stored place -> items), biggest area first:
    [{"name": "India", "count": 14, "depth": 0}, {"name": "Goa, India", ...}, {"name": "Calangute, Goa, India", ...}]."""
    agg: dict[str, int] = {}
    for place, n in counts.items():
        parts = place_parts(place)
        for i in range(len(parts)):
            key = ", ".join(parts[i:])
            agg[key] = agg.get(key, 0) + n
    order = sorted(agg, key=lambda k: [fold(p) for p in reversed(place_parts(k))])
    return [{"name": k, "count": agg[k], "depth": len(place_parts(k)) - 1} for k in order]


def places_within(name: str, known: list[str]) -> list[str]:
    """The stored places in or equal to `name`: "Goa, India" -> ["Calangute, Goa, India", "Goa, India", ...]."""
    return sorted(p for p in known if p == name or p.endswith(", " + name))
