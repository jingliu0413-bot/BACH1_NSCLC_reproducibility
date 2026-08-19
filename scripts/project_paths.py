"""Portable project paths shared by the released analysis scripts."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("NSCLC_PROJECT_ROOT", REPOSITORY_ROOT)).expanduser().resolve()
DATA_DIR = Path(os.environ.get("NSCLC_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("NSCLC_OUTPUT_DIR", PROJECT_ROOT / "out")).expanduser().resolve()
PROJECT_RESOURCES_DIR = Path(
    os.environ.get("NSCLC_RESOURCES_DIR", PROJECT_ROOT / "resources")
).expanduser().resolve()
