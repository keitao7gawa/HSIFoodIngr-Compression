from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

from ..utils import get_logger
from .dataverse_client import DataverseClient, DataverseClientConfig
from tqdm import tqdm

logger = get_logger(__name__)


@dataclass(frozen=True)
class DownloadOptions:
    output_dir: Path
    api_key: Optional[str] = None
    base_url: str = "https://dataverse.harvard.edu"
    persistent_id: str = "doi:10.7910/DVN/E7WDNQ"
    resume: bool = True
    force: bool = False


def download_dataset(options: DownloadOptions) -> Path:
    """Download dataset by iterating files individually into output_dir.

    Returns a marker file path indicating completion of listing (not a zip).
    """
    options.output_dir.mkdir(parents=True, exist_ok=True)
    client = DataverseClient(DataverseClientConfig(base_url=options.base_url, api_key=options.api_key))

    # Get file list
    files: List[Dict] = client.get_file_list(options.persistent_id)
    if not files:
        logger.warning("No files returned by dataset listing")
        return options.output_dir / ".download_empty"

    # Loop with a file-level progress bar
    for meta in tqdm(files, desc="Files", unit="file"):
        file_id = int(meta["id"])
        rel_dir = meta.get("directoryLabel") or ""
        filename = str(meta.get("filename"))
        filesize = meta.get("filesize")
        target_dir = options.output_dir / rel_dir if rel_dir else options.output_dir
        target_path = target_dir / filename

        if target_path.exists() and not options.force:
            continue

        client.download_file_by_id(file_id=file_id, dest_path=target_path, total_size=filesize, resume=options.resume)

    marker = options.output_dir / ".download_complete"
    marker.write_text("ok")
    return marker


def extract_zip(zip_path: Path, dest_dir: Path, overwrite: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target_path = dest_dir / member.filename
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and not overwrite:
                continue
            with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    logger.info("Extracted to %s", dest_dir)
    return dest_dir
