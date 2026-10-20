"""Build the offline world outline for the places map (frontend/src/lib/worldLand.ts) from Natural Earth.

Source: Natural Earth 1:110m land, public domain (https://www.naturalearthdata.com), v5.1.2:
    curl -LO https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_110m_land.geojson
    uv run python scripts/build_world_map.py ne_110m_land.geojson

Output: one SVG path in an equirectangular projection, x = longitude + 180 (0..360), y = 90 - latitude (0..180).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "src" / "lib" / "worldLand.ts"


def ring(coords: list) -> str:
    pts = [f"{lon + 180:.1f} {90 - lat:.1f}" for lon, lat in coords]
    return "M" + "L".join(pts) + "Z"


def main() -> int:
    data = json.loads(Path(sys.argv[1]).read_text())
    parts = []
    for f in data["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            parts += [ring(r) for r in poly]
    path = "".join(parts)
    OUT.write_text("// Natural Earth 1:110m land (public domain), from scripts/build_world_map.py. Do not edit.\n"
                   "// Equirectangular: x = longitude + 180, y = 90 - latitude.\n"
                   f"export const WORLD_LAND = '{path}'\n")
    print(f"{len(parts)} rings -> {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
