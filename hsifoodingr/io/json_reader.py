from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Polygon:
    points: List[Tuple[int, int]]  # [(x, y), ...]


@dataclass
class Ingredient:
    name: str
    polygons: List[Polygon]


@dataclass
class Annotation:
    dish: str
    ingredients: List[Ingredient]


def read_annotation(json_path: Path) -> Annotation:
    data = json.loads(Path(json_path).read_text())
    dish = str(data.get("dish", "")).strip()
    ingredients_raw = data.get("ingredients") or []
    ingredients: List[Ingredient] = []
    for item in ingredients_raw:
        name = str(item.get("name", "")).strip()
        polys_raw = item.get("polygons") or []
        polygons: List[Polygon] = []
        for poly in polys_raw:
            # support [[x,y], [x,y], ...] or {"points": [[x,y], ...]}
            pts = poly.get("points") if isinstance(poly, dict) else poly
            if not isinstance(pts, list):
                continue
            points: List[Tuple[int, int]] = []
            for pt in pts:
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    x, y = int(pt[0]), int(pt[1])
                    points.append((x, y))
            if points:
                polygons.append(Polygon(points=points))
        if name:
            ingredients.append(Ingredient(name=name, polygons=polygons))
    return Annotation(dish=dish, ingredients=ingredients)
