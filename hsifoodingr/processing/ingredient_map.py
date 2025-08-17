from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

from ..utils import get_logger

logger = get_logger(__name__)


def _extract_ingredient_names(json_paths: Iterable[Path]) -> Set[str]:
    names: Set[str] = set()
    for p in json_paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to read JSON %s: %s", p, e)
            continue
        # Expected structure: { dish: str, ingredients: [ {name: str, polygons: [...]}, ...] }
        ingredients = data.get("ingredients") or []
        for item in ingredients:
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return names


def _collect_json_files(raw_dir: Path) -> List[Path]:
    return [p for p in raw_dir.rglob("*.json") if p.is_file()]


def build_ingredient_map(raw_dir: Path, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_files = _collect_json_files(raw_dir)
    names = sorted(_extract_ingredient_names(json_files), key=lambda s: s.lower())

    # Ensure background class at 0
    mapping: Dict[str, int] = {"background": 0}
    next_id = 1
    for name in names:
        if name.lower() == "background":
            continue
        mapping[name] = next_id
        next_id += 1

    out_path = artifacts_dir / "ingredient_map.json"
    out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    logger.info("Ingredient map written: %s (num=%d)", out_path, len(mapping))
    return out_path
