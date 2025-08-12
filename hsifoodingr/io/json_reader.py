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


def _parse_standard_annotation(data: Dict) -> Annotation:
    dish = str(data.get("dish", "")).strip()
    ingredients_raw = data.get("ingredients") or []
    ingredients: List[Ingredient] = []
    for item in ingredients_raw:
        name = str(item.get("name", "")).strip()
        polys_raw = item.get("polygons") or []
        polygons: List[Polygon] = []
        for poly in polys_raw:
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


def _parse_geojson_like(data: Dict) -> Annotation:
    # Accept either top-level features or under markResult.features
    features = []
    if isinstance(data.get("features"), list):
        features = data.get("features")  # type: ignore
    elif isinstance(data.get("markResult"), dict) and isinstance(data["markResult"].get("features"), list):
        features = data["markResult"]["features"]  # type: ignore

    ingredients_by_name: Dict[str, List[Polygon]] = {}
    for feat in features or []:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties", {}) or {}
        # label may live under properties.content.label, properties.label, or properties.name
        label = None
        content = props.get("content") if isinstance(props, dict) else None
        if isinstance(content, dict):
            # common schema has {"label": "Sandwich/Bread", ...}
            cand = content.get("label")
            if isinstance(cand, str) and cand.strip():
                label = cand.strip()
        if not label and isinstance(props, dict):
            for key in ("label", "name", "class"):
                cand = props.get(key)
                if isinstance(cand, str) and cand.strip():
                    label = cand.strip()
                    break
        if not label:
            continue

        geom = feat.get("geometry", {}) or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Polygon" and isinstance(coords, list) and coords:
            # GeoJSON polygon: list of linear rings; use exterior ring at index 0
            exterior = coords[0]
            if isinstance(exterior, list):
                pts: List[Tuple[int, int]] = []
                for pt in exterior:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        x, y = int(round(pt[0])), int(round(pt[1]))
                        pts.append((x, y))
                if len(pts) >= 3:
                    ingredients_by_name.setdefault(label, []).append(Polygon(points=pts))
        elif gtype == "MultiPolygon" and isinstance(coords, list):
            for poly in coords:
                if isinstance(poly, list) and poly:
                    exterior = poly[0]
                    if isinstance(exterior, list):
                        pts: List[Tuple[int, int]] = []
                        for pt in exterior:
                            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                                x, y = int(round(pt[0])), int(round(pt[1]))
                                pts.append((x, y))
                        if len(pts) >= 3:
                            ingredients_by_name.setdefault(label, []).append(Polygon(points=pts))

    ingredients: List[Ingredient] = [
        Ingredient(name=name, polygons=polys) for name, polys in ingredients_by_name.items()
    ]
    dish = str(data.get("dish", "")).strip() if isinstance(data.get("dish"), str) else ""
    return Annotation(dish=dish, ingredients=ingredients)


def read_annotation(json_path: Path) -> Annotation:
    data = json.loads(Path(json_path).read_text())
    # Prefer standard schema; fallback to GeoJSON-like schema
    if isinstance(data, dict) and (data.get("ingredients") is not None):
        return _parse_standard_annotation(data)
    else:
        return _parse_geojson_like(data)
