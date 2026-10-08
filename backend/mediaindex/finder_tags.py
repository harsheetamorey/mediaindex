"""macOS Finder tags for files MediaIndex creates (export copies and clips). Never applied to originals."""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

TAGS_XATTR = "com.apple.metadata:_kMDItemUserTags"


def set_finder_tags(path: Path, tags: list[str]) -> bool:
    """Replace the Finder tags of a file this app just wrote. Best effort: returns False if unsupported or failed."""
    if sys.platform != "darwin":
        return False
    names = [t.replace("\n", " ").strip() for t in tags if t and t.strip()]
    data = plistlib.dumps(names, fmt=plistlib.FMT_BINARY).hex()
    try:
        r = subprocess.run(["/usr/bin/xattr", "-wx", TAGS_XATTR, data, str(path)], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def read_finder_tags(path: Path) -> list[str]:
    try:
        r = subprocess.run(["/usr/bin/xattr", "-px", TAGS_XATTR, str(path)], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    return [t.split("\n")[0] for t in plistlib.loads(bytes.fromhex("".join(r.stdout.split())))]
