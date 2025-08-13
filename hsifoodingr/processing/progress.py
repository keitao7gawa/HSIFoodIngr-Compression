from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py

from .manifest import _group_by_basename

REQUIRED_SUFFIXES = {".hdr", ".dat", ".png", ".json"}


@dataclass
class Progress:
    total_basenames: int
    complete_total: int
    processed: int
    ready_unprocessed: int
    incomplete: int
    processed_basenames: List[str]
    ready_unprocessed_basenames: List[str]
    incomplete_basenames: List[str]
    missing_by_basename: Dict[str, List[str]]


def _read_processed_basenames(h5_path: Path) -> Set[str]:
    if not Path(h5_path).exists():
        return set()
    with h5py.File(h5_path, "r") as f:
        if "metadata/image_basenames" not in f:
            return set()
        ds = f["metadata/image_basenames"]
        if ds.shape[0] == 0:
            return set()
        try:
            values = ds.asstr()[...].tolist()
        except Exception:
            values = [
                v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                for v in ds[...].tolist()
            ]
        return set(values)


def compute_progress(raw_dir: Path, h5_path: Path) -> Progress:
    raw_dir = Path(raw_dir)
    files = [p for p in raw_dir.rglob("*") if p.is_file()]
    groups = _group_by_basename(files)

    processed_set = _read_processed_basenames(h5_path)

    processed_basenames: List[str] = []
    ready_unprocessed_basenames: List[str] = []
    incomplete_basenames: List[str] = []
    missing_by_basename: Dict[str, List[str]] = {}

    for basename, paths in groups.items():
        suffixes = {p.suffix.lower() for p in paths}
        is_complete = REQUIRED_SUFFIXES.issubset(suffixes)
        if basename in processed_set:
            processed_basenames.append(basename)
        elif is_complete:
            ready_unprocessed_basenames.append(basename)
        else:
            incomplete_basenames.append(basename)
            missing = [s for s in sorted(REQUIRED_SUFFIXES) if s not in suffixes]
            if missing:
                missing_by_basename[basename] = missing

    processed_basenames.sort()
    ready_unprocessed_basenames.sort()
    incomplete_basenames.sort()

    return Progress(
        total_basenames=len(groups),
        complete_total=len([1 for b, ps in groups.items() if REQUIRED_SUFFIXES.issubset({p.suffix.lower() for p in ps})]),
        processed=len(processed_basenames),
        ready_unprocessed=len(ready_unprocessed_basenames),
        incomplete=len(incomplete_basenames),
        processed_basenames=processed_basenames,
        ready_unprocessed_basenames=ready_unprocessed_basenames,
        incomplete_basenames=incomplete_basenames,
        missing_by_basename=missing_by_basename,
    )
