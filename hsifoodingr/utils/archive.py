from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path
from typing import Literal

from .logging import get_logger

logger = get_logger(__name__)

ArchiveType = Literal["zip", "tar", "targz", "tarxz", "unknown"]


def _remove_all_suffixes(name: str) -> str:
    stem = name
    # remove multi-part suffixes like .zip.tar.gz iteratively
    while True:
        base, ext = os.path.splitext(stem)
        if not ext:
            break
        stem = base
    return stem


def detect_archive_type(path: Path) -> ArchiveType:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "targz"
    if name.endswith(".tar.xz") or name.endswith(".txz"):
        return "tarxz"
    if name.endswith(".tar"):
        return "tar"
    # compound like .zip.tar.gz will be handled iteratively in safe_extract
    if ".tar" in name or ".zip" in name:
        return "unknown"
    return "unknown"


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve(strict=False)
        target = target.resolve(strict=False)
    except Exception:
        # best effort; fallback to string check
        pass
    return os.path.commonpath([str(directory)]) == os.path.commonpath([str(directory), str(target)])


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    for member in tar.getmembers():
        member_path = dest / member.name
        if not _is_within_directory(dest, member_path):
            raise RuntimeError(f"Blocked path traversal: {member.name}")
    tar.extractall(dest)  # safe by pre-check


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    for zi in zf.infolist():
        target = dest / zi.filename
        if not _is_within_directory(dest, target):
            raise RuntimeError(f"Blocked path traversal: {zi.filename}")
        if zi.is_dir():
            (dest / zi.filename).mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(zi, "r") as src, open(target, "wb") as dst:
                dst.write(src.read())


def _find_nested_archive(root: Path) -> Path | None:
    """Find a nested archive within root; prefer .zip, then tar variants."""
    candidates = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".zip"):
            candidates.append((0, p))
        elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz")):
            candidates.append((1, p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], len(str(t[1]))))
    return candidates[0][1]


def safe_extract(archive_path: Path, dest_root: Path) -> Path:
    """Safely extract archives, supporting nested multi-extensions.

    Returns the final extracted directory root.
    """
    archive_path = Path(archive_path)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    # Create a working folder based on the full filename, removing all suffixes
    final_dir_name = _remove_all_suffixes(archive_path.name)
    work_dir = dest_root / final_dir_name
    work_dir.mkdir(parents=True, exist_ok=True)

    current_input = archive_path
    current_output = work_dir

    # Iterate while we keep discovering nested archives inside the output
    for _ in range(5):
        lower = current_input.name.lower()
        if lower.endswith(".zip"):
            with zipfile.ZipFile(current_input, "r") as zf:
                _safe_extract_zip(zf, current_output)
        elif lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz")):
            mode = "r:"
            if lower.endswith((".tar.gz", ".tgz")):
                mode = "r:gz"
            elif lower.endswith((".tar.xz", ".txz")):
                mode = "r:xz"
            with tarfile.open(current_input, mode) as tf:
                _safe_extract_tar(tf, current_output)
        else:
            # Not a recognized archive extension; stop
            break

        # After extraction, look for a nested archive to continue
        nested = _find_nested_archive(current_output)
        if nested is None:
            break
        current_input = nested
        # keep output as current_output

    logger.info("Extracted %s to %s", archive_path, work_dir)
    return work_dir


def cleanup_path(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
        elif path.is_dir():
            # remove tree
            for p in sorted(path.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    try:
                        p.unlink()
                    except Exception:
                        pass
                elif p.is_dir():
                    try:
                        p.rmdir()
                    except Exception:
                        pass
            try:
                path.rmdir()
            except Exception:
                pass
    except Exception:
        # non-fatal
        logger.warning("Failed to cleanup %s", path)
