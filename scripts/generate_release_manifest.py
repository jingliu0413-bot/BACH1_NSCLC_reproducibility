"""Generate SHA256 checksums for the lightweight repository contents."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", "__pycache__"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path == OUTPUT or EXCLUDED_PARTS.intersection(relative.parts):
            continue
        files.append(relative)
    lines = [f"{digest(ROOT / path)}  {path.as_posix()}" for path in sorted(files)]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {OUTPUT} with {len(lines)} entries")


if __name__ == "__main__":
    main()
