"""Build the bundled offline place list (backend/mediaindex/resources/places.tsv.gz) from GeoNames.

Source: GeoNames cities15000 (every place with 15,000+ people), admin1CodesASCII and countryInfo, CC BY 4.0
(https://www.geonames.org). Download them first, then run:

    curl -LO https://download.geonames.org/export/dump/cities15000.zip && unzip cities15000.zip
    curl -LO https://download.geonames.org/export/dump/admin1CodesASCII.txt
    curl -LO https://download.geonames.org/export/dump/countryInfo.txt
    uv run python scripts/build_places.py DOWNLOAD_DIR

Output columns: name, region (admin1), country, latitude, longitude, population. Sections of a city (PPLX, e.g.
"Paris 16 Passy") and historical or abandoned places are left out. MediaIndex only reads this file; it never looks
places up online.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

SKIP_CODES = {"PPLX", "PPLH", "PPLQ", "PPLW"}  # section of a city, historical, abandoned, destroyed
OUT = Path(__file__).resolve().parents[1] / "backend" / "mediaindex" / "resources" / "places.tsv.gz"


def main() -> int:
    src = Path(sys.argv[1])
    countries = {}
    for line in (src / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            f = line.split("\t")
            countries[f[0]] = f[4]
    regions = {}
    for line in (src / "admin1CodesASCII.txt").read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        regions[f[0]] = f[1]
    rows = []
    for line in (src / "cities15000.txt").read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        name, lat, lon, code, cc, a1, pop = f[1], float(f[4]), float(f[5]), f[7], f[8], f[10], f[14]
        if code in SKIP_CODES:
            continue
        rows.append((name, regions.get(f"{cc}.{a1}", ""), countries.get(cc, cc), f"{lat:.4f}", f"{lon:.4f}",
                     str(int(pop or 0))))
    rows.sort(key=lambda r: (r[2], r[1], r[0]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(OUT, "wb", compresslevel=9, mtime=0) as g:  # mtime=0: the same input gives the same file
        g.write(("\n".join("\t".join(r) for r in rows) + "\n").encode("utf-8"))
    print(f"{len(rows)} places -> {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
