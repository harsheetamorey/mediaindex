"""Filesystem safety: explicit roots, no traversal, no symlink escapes."""

from __future__ import annotations

import os
from pathlib import Path


class PathRejected(ValueError):
    pass


def validate_root(path: str) -> Path:
    """Validate a user-selected library root. Refuses '/', the home directory itself and non-directories."""
    if not path or not os.path.isabs(os.path.expanduser(path)):
        raise PathRejected("library folder must be an absolute path")
    p = Path(os.path.expanduser(path)).resolve()
    if not p.is_dir():
        raise PathRejected(f"not a directory: {p}")
    if p == Path(p.anchor) or p == Path.home().resolve():
        raise PathRejected("choose a specific media folder, not the filesystem root or your whole home directory")
    return p


def is_within(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_in_root(root: Path | str, rel_path: str) -> Path:
    """Resolve a stored root-relative path, rejecting traversal and symlinks that escape the root."""
    root = Path(root).resolve()
    if os.path.isabs(rel_path) or ".." in Path(rel_path).parts:
        raise PathRejected("invalid relative path")
    target = (root / rel_path).resolve()
    if not is_within(target, root):
        raise PathRejected("path escapes library root")
    return target


def walk_files(root: Path, extensions: set[str], skipped: list[tuple[str, str]] | None = None):
    """Yield (abs_path, rel_path) for files under root with allowed extensions.

    Does not descend into symlinked directories; symlinked files are only accepted if their
    target is inside the root. Hidden files and directories are skipped.
    """
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if Path(name).suffix.lower() not in extensions:
                continue
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            if p.is_symlink():
                target = p.resolve()
                if not is_within(target, root):
                    if skipped is not None:
                        skipped.append((rel, "symlink points outside library root"))
                    continue
            if not p.is_file():
                continue
            yield p, rel
