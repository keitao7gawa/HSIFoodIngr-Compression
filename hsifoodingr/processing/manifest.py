from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SampleFiles:
    basename: str
    hdr: Path
    dat: Path
    png: Path
    json_path: Path


REQUIRED_SUFFIXES = {".hdr", ".dat", ".png", ".json"}


def _group_by_basename(files: Iterable[Path]) -> Dict[str, Set[Path]]:
    groups: Dict[str, Set[Path]] = {}
    for p in files:
        if not p.is_file():
            continue
        stem = p.stem  # name without suffix
        groups.setdefault(stem, set()).add(p)
    return groups


def _is_complete_group(paths: Set[Path]) -> bool:
    suffixes = {p.suffix.lower() for p in paths}
    return REQUIRED_SUFFIXES.issubset(suffixes)


def scan_raw(raw_dir: Path) -> Tuple[List[Path], List[SampleFiles]]:
    """Scan raw_dir recursively and return all files, and complete sample groups."""
    all_files: List[Path] = [p for p in raw_dir.rglob("*") if p.is_file()]
    groups = _group_by_basename(all_files)
    complete_samples: List[SampleFiles] = []
    for basename, paths in groups.items():
        if _is_complete_group(paths):
            # map by suffix
            mapping = {p.suffix.lower(): p for p in paths}
            complete_samples.append(
                SampleFiles(
                    basename=basename,
                    hdr=mapping[".hdr"],
                    dat=mapping[".dat"],
                    png=mapping[".png"],
                    json_path=mapping[".json"],
                )
            )
    complete_samples.sort(key=lambda s: s.basename)
    return all_files, complete_samples


def write_manifest(artifacts_dir: Path, all_files: List[Path], complete_samples: List[SampleFiles]) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts_dir / "file_manifest.txt"
    processable_path = artifacts_dir / "processable.txt"
    json_path = artifacts_dir / "processable.json"

    manifest_path.write_text("\n".join(str(p) for p in sorted(all_files)))
    processable_path.write_text("\n".join(s.basename for s in complete_samples))

    json_path.write_text(
        json.dumps(
            [
                {
                    "basename": s.basename,
                    "hdr": str(s.hdr),
                    "dat": str(s.dat),
                    "png": str(s.png),
                    "json": str(s.json_path),
                }
                for s in complete_samples
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    logger.info(
        "Manifest written: %s (all), %s (processable %d), %s (details)",
        manifest_path,
        processable_path,
        len(complete_samples),
        json_path,
    )


def build_manifest(raw_dir: Path, artifacts_dir: Path) -> None:
    all_files, complete = scan_raw(raw_dir)
    write_manifest(artifacts_dir, all_files, complete)
