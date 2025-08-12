from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..io.json_reader import Annotation, Ingredient, Polygon


def _clip_point(pt: Tuple[int, int], width: int, height: int) -> Tuple[int, int]:
    x, y = pt
    x = max(0, min(width - 1, int(x)))
    y = max(0, min(height - 1, int(y)))
    return x, y


def _polygon_points(poly: Polygon, width: int, height: int) -> List[Tuple[int, int]]:
    return [_clip_point((x, y), width, height) for (x, y) in poly.points]


def build_mask(
    annotation: Annotation,
    ingredient_name_to_id: Dict[str, int],
    image_size_hw: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Build uint8 mask (H, W) from annotation polygons using provided ingredient map.

    - Later polygons overwrite earlier ones ("last write wins").
    - Ingredients are processed in the order they appear in the annotation.
    - Unknown ingredient names are ignored.
    """
    height, width = image_size_hw
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for ingredient in annotation.ingredients:
        if not isinstance(ingredient, Ingredient):
            continue
        name = ingredient.name
        if not name or name not in ingredient_name_to_id:
            continue
        value = int(ingredient_name_to_id[name])
        if value < 0 or value > 255:
            value = int(value % 256)
        for poly in ingredient.polygons:
            if not isinstance(poly, Polygon) or len(poly.points) < 3:
                continue
            pts = _polygon_points(poly, width, height)
            if len(pts) < 3:
                continue
            draw.polygon(pts, outline=value, fill=value)

    mask = np.asarray(mask_img, dtype=np.uint8)
    return mask


def load_ingredient_map(path: Path) -> Dict[str, int]:
    import json

    data = json.loads(Path(path).read_text())
    mapping: Dict[str, int] = {}
    for k, v in data.items():
        try:
            mapping[str(k)] = int(v)
        except Exception:
            continue
    return mapping
