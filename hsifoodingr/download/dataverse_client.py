from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import requests
from requests import Response
from tqdm import tqdm

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DataverseClientConfig:
    base_url: str = "https://dataverse.harvard.edu"
    api_key: Optional[str] = None
    timeout_sec: int = 60
    chunk_size: int = 1024 * 1024  # 1 MiB


class DataverseClient:
    def __init__(self, config: DataverseClientConfig) -> None:
        self.config = config
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({"X-Dataverse-key": config.api_key})

    def _dataset_zip_url(self, persistent_id: str) -> str:
        return f"{self.config.base_url}/api/access/dataset/:persistentId?persistentId={persistent_id}"

    def _dataset_latest_version_url(self, persistent_id: str) -> str:
        return f"{self.config.base_url}/api/datasets/:persistentId/versions/:latest?persistentId={persistent_id}"

    def _datafile_url(self, file_id: int) -> str:
        return f"{self.config.base_url}/api/access/datafile/{file_id}"

    def get_file_list(self, persistent_id: str) -> List[Dict]:
        """Return a list of file metadata dicts with keys: id, filename, directoryLabel, filesize."""
        url = self._dataset_latest_version_url(persistent_id)
        r = self.session.get(url, timeout=self.config.timeout_sec)
        r.raise_for_status()
        data = r.json()
        files = data.get("data", {}).get("files", []) if isinstance(data, dict) else []
        results: List[Dict] = []
        for f in files:
            df = f.get("dataFile", {}) if isinstance(f, dict) else {}
            if not isinstance(df, dict):
                continue
            file_id = df.get("id")
            filename = df.get("filename")
            filesize = df.get("filesize")
            directory_label = f.get("directoryLabel") if isinstance(f, dict) else None
            if file_id is None or filename is None:
                continue
            results.append(
                {
                    "id": int(file_id),
                    "filename": str(filename),
                    "filesize": int(filesize) if isinstance(filesize, int) else None,
                    "directoryLabel": str(directory_label) if isinstance(directory_label, str) else "",
                }
            )
        return results

    def download_dataset_zip(
        self,
        persistent_id: str,
        dest_path: Path,
        resume: bool = True,
    ) -> Path:
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine resume position
        resume_from = 0
        if resume and dest_path.exists():
            resume_from = dest_path.stat().st_size

        url = self._dataset_zip_url(persistent_id)
        headers: Dict[str, str] = {}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        logger.info("Starting download: %s", url)
        with self.session.get(url, stream=True, headers=headers, timeout=self.config.timeout_sec) as r:
            r.raise_for_status()
            # Try to determine total size for progress
            total_size = None
            if resume_from > 0:
                # If server supports partial content, Content-Range is present
                if r.status_code == 206:
                    content_range = r.headers.get("Content-Range")
                    if content_range and "/" in content_range:
                        try:
                            total_size = int(content_range.split("/")[-1])
                        except Exception:
                            total_size = None
                else:
                    # Server did not honor range; restart from scratch
                    resume_from = 0
            if total_size is None:
                try:
                    total_size = int(r.headers.get("Content-Length", "0"))
                except Exception:
                    total_size = None

            mode = "ab" if resume_from > 0 else "wb"
            # Configure progress bar format with percentage when total is known
            has_total = bool(total_size and total_size >= resume_from)
            bar_format = (
                "{l_bar}{bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ) if has_total else None
            with open(dest_path, mode) as f, tqdm(
                total=total_size if has_total else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                initial=resume_from if (has_total and resume_from > 0) else 0,
                desc=f"Downloading {dest_path.name}",
                leave=True,
                dynamic_ncols=True,
                bar_format=bar_format,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=self.config.chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info("Download complete: %s (%s bytes)", dest_path, dest_path.stat().st_size)
        return dest_path

    def download_file_by_id(self, file_id: int, dest_path: Path, total_size: Optional[int] = None, resume: bool = True) -> Path:
        """Download a single datafile by ID with progress and resume support."""
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        resume_from = 0
        if resume and dest_path.exists():
            resume_from = dest_path.stat().st_size

        url = self._datafile_url(file_id)
        headers: Dict[str, str] = {}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        with self.session.get(url, stream=True, headers=headers, timeout=self.config.timeout_sec) as r:
            r.raise_for_status()
            # Determine size
            size = total_size
            if resume_from > 0 and r.status_code != 206:
                # Server ignored range; restart
                resume_from = 0
            if size is None:
                try:
                    size = int(r.headers.get("Content-Length", "0"))
                except Exception:
                    size = None

            has_total = bool(size and size >= resume_from)
            bar_format = (
                "{l_bar}{bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ) if has_total else None
            mode = "ab" if resume_from > 0 else "wb"
            with open(dest_path, mode) as f, tqdm(
                total=size if has_total else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                initial=resume_from if (has_total and resume_from > 0) else 0,
                desc=f"{dest_path.name}",
                leave=False,
                dynamic_ncols=True,
                bar_format=bar_format,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=self.config.chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
        return dest_path
