"""Generate SHA256 checksums for lightweight Git-tracked release contents."""

from __future__ import annotations

import hashlib
import subprocess
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
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for line in result.stdout.splitlines():
        relative = Path(line)
        path = ROOT / relative
        if not path.is_file() or path == OUTPUT or EXCLUDED_PARTS.intersection(relative.parts):
            continue
        files.append(relative)
    lines = [f"{digest(ROOT / path)}  {path.as_posix()}" for path in sorted(files)]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {OUTPUT} with {len(lines)} entries")


if __name__ == "__main__":
    main()
