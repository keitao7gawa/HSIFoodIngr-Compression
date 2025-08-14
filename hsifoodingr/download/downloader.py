from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils import get_logger
from .dataverse_client import DataverseClient, DataverseClientConfig

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
    options.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = options.output_dir / "HSIFoodIngr-64.zip"

    if zip_path.exists() and not options.force and not options.resume:
        logger.info("ZIP already exists and neither --force nor --resume specified: %s", zip_path)
        return zip_path

    client = DataverseClient(
        DataverseClientConfig(base_url=options.base_url, api_key=options.api_key)
    )
    return client.download_dataset_zip(options.persistent_id, zip_path, resume=options.resume)


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
