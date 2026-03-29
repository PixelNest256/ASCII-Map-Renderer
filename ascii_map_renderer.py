#!/usr/bin/env python3
"""Render prefecture, city, and ward boundaries with rain radar as terminal text art."""

from __future__ import annotations
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
from shutil import get_terminal_size
import string
import sys
import time
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import requests as requests_pkg
    from PIL import ImageDraw as PILImageDraw_pkg
    from PIL import ImageFont as PILImageFont_pkg
    from PIL.Image import Image as PILImageType

# 実行時に遅延ロードするモジュール(型はAnyにしてOptional member accessを抑制)
np: Any = None
requests: Any = None
Image: Any = None
ImageDraw: Any = None
ImageFont: Any = None


def load_runtime_dependencies() -> None:
    global np, requests, Image, ImageDraw, ImageFont

    if all(module is not None for module in (np, requests, Image, ImageDraw, ImageFont)):
        return

    try:
        import numpy as numpy_module
        import requests as requests_module
        from PIL import Image as Image_module
        from PIL import ImageDraw as ImageDraw_module
        from PIL import ImageFont as ImageFont_module
    except ImportError as exc:  # pragma: no cover - exercised only when deps are missing
        raise RuntimeError(
            "Missing runtime dependencies. Install with: pip install pillow numpy requests"
        ) from exc

    np = numpy_module
    requests = requests_module
    Image = Image_module
    ImageDraw = ImageDraw_module
    ImageFont = ImageFont_module


DEFAULT_LAT = 35.681236
DEFAULT_LON = 139.767125
DEFAULT_MARKER_LAT = DEFAULT_LAT
DEFAULT_MARKER_LON = DEFAULT_LON
DEFAULT_SCALE_METERS = 150.0
CENTER_RAIN_RADAR = True
JST = ZoneInfo("Asia/Tokyo")
CACHE_DIRNAME = "cache"
USER_AGENT = "AsciiMapRenderer/1.0"
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_ENDPOINTS = [
    OVERPASS_ENDPOINT,
    "https://overpass.kumi.systems/api/interpreter",
]

PREF_STYLE = "\x1b[1;97m"
CITY_STYLE = "\x1b[37m"
WARD_STYLE = "\x1b[90m"
MARKER_STYLE = "\x1b[1;92m"
RESET_STYLE = "\x1b[0m"
RESET_FG_STYLE = "\x1b[39m"
RESET_BG_STYLE = "\x1b[49m"

TILE_W = 18
TILE_H = 28
LINE_WIDTH = 2
ALLOWED_ADMIN_LEVELS = {4, 7, 8}
CACHE_VERSION = "v3"
RADAR_ZOOM = 10
RADAR_TARGET_URL = "https://www.jma.go.jp/bosai/jmatile/data/nowc/targetTimes_N1.json"
RADAR_TILE_URL = "https://www.jma.go.jp/bosai/jmatile/data/nowc/{basetime}/none/{validtime}/surf/hrpns/{zoom}/{x}/{y}.png"
RADAR_BASE_RGB = (12, 14, 18)
RADAR_BLEND = 0.55
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
DEFAULT_CELL_ASPECT = 2.0

Point = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Ring = List[Point]
Rings = List[Ring]


@dataclass(frozen=True)
class LayerSpec:
    name: str
    style: str
    priority: int


@dataclass(frozen=True)
class BoundaryFeature:
    osm_type: str
    osm_id: int
    admin_level: int | None
    name: str
    style: str
    priority: int
    lines: List[List[Point]]


@dataclass(frozen=True)
class RadarCellSample:
    tile_x: int
    tile_y: int
    px: int
    py: int


@dataclass(frozen=True)
class RadarSampling:
    cells: List[List[RadarCellSample]]
    tile_bounds: Tuple[int, int, int, int]


@dataclass
class RadarState:
    target: Tuple[str, str] | None = None
    tiles: Dict[Tuple[int, int], "PILImageType"] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticScene:
    chars: List[List[str]]
    styles: List[List[str]]
    location_label: str
    radar_sampling: RadarSampling


@dataclass(frozen=True)
class CurrentWeatherSnapshot:
    weather_label: str
    precipitation_mm: float
    wind_direction_label: str
    wind_speed_ms: float
    updated_at: datetime


def build_session() -> "requests_pkg.Session":
    load_runtime_dependencies()
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "ja,en;q=0.8",
        }
    )
    return session


def project_to_meters(lat: float, lon: float, lat0: float, lon0: float) -> Point:
    """Project WGS84 geodetic coordinates to local ENU meters around lat0/lon0."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref = _reference_enu(lat0, lon0)
    x, y, z = _ecef_from_geodetic(lat_rad, lon_rad, 0.0)
    dx = x - ref[0]
    dy = y - ref[1]
    dz = z - ref[2]
    sin_lat = ref[3]
    cos_lat = ref[4]
    sin_lon = ref[5]
    cos_lon = ref[6]
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    return east, north


@lru_cache(maxsize=64)
def _reference_enu(lat0: float, lon0: float) -> Tuple[float, float, float, float, float, float, float]:
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    x0, y0, z0 = _ecef_from_geodetic(lat0_rad, lon0_rad, 0.0)
    sin_lat = math.sin(lat0_rad)
    cos_lat = math.cos(lat0_rad)
    sin_lon = math.sin(lon0_rad)
    cos_lon = math.cos(lon0_rad)
    return x0, y0, z0, sin_lat, cos_lat, sin_lon, cos_lon


def _ecef_from_geodetic(lat_rad: float, lon_rad: float, height_m: float) -> Vec3:
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + height_m) * cos_lat * cos_lon
    y = (n + height_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + height_m) * sin_lat
    return x, y, z


def detect_terminal_cell_aspect() -> float:
    """Estimate the terminal cell height/width ratio on Windows."""
    if os.name != "nt":
        return DEFAULT_CELL_ASPECT

    try:
        import ctypes
        from ctypes import wintypes

        class COORD(ctypes.Structure):
            _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

        class CONSOLE_FONT_INFOEX(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.ULONG),
                ("nFont", wintypes.DWORD),
                ("dwFontSize", COORD),
                ("FontFamily", wintypes.UINT),
                ("FontWeight", wintypes.UINT),
                ("FaceName", ctypes.c_wchar * 32),
            ]

        kernel32 = ctypes.windll.kernel32
        get_std_handle = kernel32.GetStdHandle
        get_std_handle.argtypes = [wintypes.DWORD]
        get_std_handle.restype = wintypes.HANDLE
        get_font = kernel32.GetCurrentConsoleFontEx
        get_font.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(CONSOLE_FONT_INFOEX)]
        get_font.restype = wintypes.BOOL

        handle = get_std_handle(-11)
        if not handle:
            return DEFAULT_CELL_ASPECT

        info = CONSOLE_FONT_INFOEX()
        info.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
        if not get_font(handle, False, ctypes.byref(info)):
            return DEFAULT_CELL_ASPECT

        width = max(1, int(info.dwFontSize.X))
        height = max(1, int(info.dwFontSize.Y))
        aspect = height / width
        if not math.isfinite(aspect) or aspect <= 0.0:
            return DEFAULT_CELL_ASPECT
        return aspect
    except Exception:
        return DEFAULT_CELL_ASPECT


def _geodetic_from_ecef(x: float, y: float, z: float) -> Point:
    """Convert WGS84 ECEF coordinates back to latitude/longitude."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p == 0.0:
        lat = math.copysign(math.pi / 2.0, z)
        return math.degrees(lat), math.degrees(lon)

    b = WGS84_A * (1.0 - WGS84_F)
    ep2 = (WGS84_A * WGS84_A - b * b) / (b * b)
    theta = math.atan2(z * WGS84_A, p * b)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    lat = math.atan2(
        z + ep2 * b * sin_theta * sin_theta * sin_theta,
        p - WGS84_E2 * WGS84_A * cos_theta * cos_theta * cos_theta,
    )
    return math.degrees(lat), math.degrees(lon)


def _ecef_from_enu(east: float, north: float, up: float, reference: Tuple[float, float, float, float, float, float, float]) -> Vec3:
    x0, y0, z0, sin_lat, cos_lat, sin_lon, cos_lon = reference
    x = x0 - sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    y = y0 + cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    z = z0 + cos_lat * north + sin_lat * up
    return x, y, z


def meters_to_latlon(east: float, north: float, lat0: float, lon0: float) -> Point:
    reference = _reference_enu(lat0, lon0)
    x, y, z = _ecef_from_enu(east, north, 0.0, reference)
    return _geodetic_from_ecef(x, y, z)


def ensure_cache_dir(base_dir: Path) -> Path:
    cache_dir = base_dir / CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def viewport_bounds(cols: int, rows: int, scale_m: float, ground_aspect: float) -> Tuple[float, float, float, float]:
    half_w = cols * scale_m / 2.0
    half_h = rows * scale_m * ground_aspect / 2.0
    return -half_w, half_w, -half_h, half_h


def viewport_bbox(
    lat: float,
    lon: float,
    cols: int,
    rows: int,
    scale_m: float,
    ground_aspect: float,
    padding_cells: float = 1.0,
) -> Tuple[float, float, float, float]:
    half_w = (cols / 2.0 + padding_cells) * scale_m
    half_h = (rows / 2.0 + padding_cells) * scale_m * ground_aspect
    corners = [
        meters_to_latlon(-half_w, -half_h, lat, lon),
        meters_to_latlon(half_w, -half_h, lat, lon),
        meters_to_latlon(-half_w, half_h, lat, lon),
        meters_to_latlon(half_w, half_h, lat, lon),
    ]
    lats = [pt[0] for pt in corners]
    lons = [pt[1] for pt in corners]
    return min(lats), min(lons), max(lats), max(lons)


def cell_center_meters(col: int, row: int, cols: int, rows: int, scale_m: float, ground_aspect: float) -> Point:
    half_w = cols * scale_m / 2.0
    half_h = rows * scale_m * ground_aspect / 2.0
    east = -half_w + (col + 0.5) * scale_m
    north = half_h - (row + 0.5) * scale_m * ground_aspect
    return east, north


def clamp_latitude(lat: float) -> float:
    return max(-85.05112878, min(85.05112878, lat))


def webmercator_tile_xy(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    lat = clamp_latitude(lat)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return x, y


def webmercator_tile_bounds_from_bbox(south: float, west: float, north: float, east: float, zoom: int) -> Tuple[int, int, int, int]:
    corners = [
        webmercator_tile_xy(north, west, zoom),
        webmercator_tile_xy(north, east, zoom),
        webmercator_tile_xy(south, west, zoom),
        webmercator_tile_xy(south, east, zoom),
    ]
    xs = [pt[0] for pt in corners]
    ys = [pt[1] for pt in corners]
    x_min = math.floor(min(xs))
    x_max = math.floor(max(xs))
    y_min = math.floor(min(ys))
    y_max = math.floor(max(ys))
    return x_min, x_max, y_min, y_max


def meters_to_canvas(
    x: float,
    y: float,
    cols: int,
    rows: int,
    scale_m: float,
    tile_h: int,
    ground_aspect: float,
) -> Tuple[float, float]:
    half_w = cols * scale_m / 2.0
    half_h = rows * scale_m * ground_aspect / 2.0
    px = (x + half_w) / scale_m * TILE_W
    py = (half_h - y) / (scale_m * ground_aspect) * tile_h
    return px, py


def meters_to_point(
    x: float,
    y: float,
    cols: int,
    rows: int,
    scale_m: float,
    tile_h: int,
    ground_aspect: float,
) -> Tuple[int, int]:
    half_w = cols * scale_m / 2.0
    half_h = rows * scale_m * ground_aspect / 2.0
    px = int(round((x + half_w) / scale_m * TILE_W))
    py = int(round((half_h - y) / (scale_m * ground_aspect) * tile_h))
    return px, py


def clip_segment_to_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Tuple[float, float, float, float] | None:
    # Cohen-Sutherland clipping in projected meters.
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def out_code(x: float, y: float) -> int:
        code = INSIDE
        if x < xmin:
            code |= LEFT
        elif x > xmax:
            code |= RIGHT
        if y < ymin:
            code |= BOTTOM
        elif y > ymax:
            code |= TOP
        return code

    code0 = out_code(x0, y0)
    code1 = out_code(x1, y1)

    while True:
        if not (code0 | code1):
            return x0, y0, x1, y1
        if code0 & code1:
            return None

        code_out = code0 or code1
        if code_out & TOP:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
            y = ymax
        elif code_out & BOTTOM:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
            y = ymin
        elif code_out & RIGHT:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
            x = xmax
        else:
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
            x = xmin

        if code_out == code0:
            x0, y0 = x, y
            code0 = out_code(x0, y0)
        else:
            x1, y1 = x, y
            code1 = out_code(x1, y1)


def rasterize_rings_to_canvas(
    rings: Rings,
    lat0: float,
    lon0: float,
    cols: int,
    rows: int,
    scale_m: float,
    tile_h: int,
    ground_aspect: float,
) -> PILImageType:
    canvas_w = cols * TILE_W
    canvas_h = rows * tile_h
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    xmin, xmax, ymin, ymax = viewport_bounds(cols, rows, scale_m, ground_aspect)

    for ring in rings:
        if len(ring) < 2:
            continue

        points_m = [project_to_meters(lat, lon, lat0, lon0) for lon, lat in ring]
        for (x0, y0), (x1, y1) in zip(points_m, points_m[1:]):
            clipped = clip_segment_to_rect(x0, y0, x1, y1, xmin, xmax, ymin, ymax)
            if clipped is None:
                continue

            cx0, cy0, cx1, cy1 = clipped
            px0, py0 = meters_to_canvas(cx0, cy0, cols, rows, scale_m, tile_h, ground_aspect)
            px1, py1 = meters_to_canvas(cx1, cy1, cols, rows, scale_m, tile_h, ground_aspect)
            draw.line((px0, py0, px1, py1), fill=255, width=LINE_WIDTH)

    return img


def draw_polyline_to_canvas(
    draw: PILImageDraw_pkg.ImageDraw,
    points: List[Point],
    lat0: float,
    lon0: float,
    cols: int,
    rows: int,
    scale_m: float,
    tile_h: int,
    ground_aspect: float,
) -> None:
    if len(points) < 2:
        return

    xmin, xmax, ymin, ymax = viewport_bounds(cols, rows, scale_m, ground_aspect)
    projected = [project_to_meters(lat, lon, lat0, lon0) for lon, lat in points]
    for (x0, y0), (x1, y1) in zip(projected, projected[1:]):
        clipped = clip_segment_to_rect(x0, y0, x1, y1, xmin, xmax, ymin, ymax)
        if clipped is None:
            continue

        cx0, cy0, cx1, cy1 = clipped
        px0, py0 = meters_to_canvas(cx0, cy0, cols, rows, scale_m, tile_h, ground_aspect)
        px1, py1 = meters_to_canvas(cx1, cy1, cols, rows, scale_m, tile_h, ground_aspect)
        draw.line((px0, py0, px1, py1), fill=255, width=LINE_WIDTH)


def rasterize_features_to_canvas(
    features: List[BoundaryFeature],
    lat0: float,
    lon0: float,
    cols: int,
    rows: int,
    scale_m: float,
    tile_h: int,
    ground_aspect: float,
) -> PILImageType:
    canvas_w = cols * TILE_W
    canvas_h = rows * tile_h
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    for feature in features:
        for line in feature.lines:
            draw_polyline_to_canvas(draw, line, lat0, lon0, cols, rows, scale_m, tile_h, ground_aspect)
    return img


def fetch_admin_boundaries_in_bbox(
    session: requests_pkg.Session,
    cache_dir: Path,
    lat: float,
    lon: float,
    cols: int,
    rows: int,
    scale_m: float,
    ground_aspect: float,
) -> List[dict]:
    south, west, north, east = viewport_bbox(lat, lon, cols, rows, scale_m, ground_aspect, padding_cells=1.5)
    cache_key = hashlib.sha1(
        f"{CACHE_VERSION}:overpass:{south:.6f},{west:.6f},{north:.6f},{east:.6f}".encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        try:
            cached = load_json(cache_path)
            if isinstance(cached, list) and cached:
                return cached
        except Exception:
            pass

    query = f"""
[out:json][timeout:120];
(
  relation["boundary"="administrative"]["admin_level"~"^(4|7|8)$"]({south},{west},{north},{east});
);
out geom;
"""
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            resp = session.get(endpoint, params={"data": query}, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            if elements:
                save_json(cache_path, elements)
                return elements
        except Exception as exc:  # noqa: BLE001 - network and decode failures are expected here
            last_error = exc
            continue

    if cache_path.exists():
        try:
            cached = load_json(cache_path)
            if isinstance(cached, list):
                return cached
        except Exception:
            pass

    if last_error is not None:
        raise RuntimeError(f"Unable to fetch administrative boundaries: {last_error}") from last_error
    return []


def parse_boundary_features(elements: List[dict]) -> List[BoundaryFeature]:
    features: List[BoundaryFeature] = []

    for element in elements:
        tags = element.get("tags", {})
        admin_level_raw = tags.get("admin_level")
        try:
            admin_level = int(admin_level_raw) if admin_level_raw is not None else None
        except (TypeError, ValueError):
            admin_level = None

        if admin_level not in ALLOWED_ADMIN_LEVELS:
            continue

        name = tags.get("name") or tags.get("name:ja") or f"{element.get('type')} {element.get('id')}"
        style, priority = style_for_admin_level(admin_level)
        lines: List[List[Point]] = []

        if element.get("type") == "relation":
            for member in element.get("members", []):
                if member.get("type") != "way":
                    continue
                if member.get("role") not in {"outer", "boundary", ""}:
                    continue
                geometry = member.get("geometry") or []
                if len(geometry) >= 2:
                    lines.append([(float(pt["lon"]), float(pt["lat"])) for pt in geometry])

        if lines:
            features.append(
                BoundaryFeature(
                    osm_type=element.get("type", "unknown"),
                    osm_id=int(element.get("id", 0)),
                    admin_level=admin_level,
                    name=name,
                    style=style,
                    priority=priority,
                    lines=lines,
                )
            )

    return features


def style_for_admin_level(admin_level: int | None) -> Tuple[str, int]:
    if admin_level is None:
        return "\x1b[2m", 0
    if admin_level == 4:
        return PREF_STYLE, 0
    if admin_level == 7:
        return CITY_STYLE, 1
    if admin_level == 8:
        return WARD_STYLE, 2
    return "\x1b[2m", 3


def resolve_address_component(address: dict, keys: List[str], fallback: str) -> str:
    for key in keys:
        value = address.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def format_location_label(pref_name: str, city_name: str, ward_name: str) -> str:
    parts = [part for part in (pref_name, city_name, ward_name) if part and part != "unknown"]
    return " ".join(parts) if parts else "不明"


def format_jst_timestamp(value: datetime) -> str:
    return value.astimezone(JST).strftime("%Y/%m/%d %H:%M:%S")


def weather_label_from_code(code: int | None) -> str:
    if code is None:
        return "不明"

    if code == 0:
        return "晴れ"
    if code == 1:
        return "薄曇り"
    if code == 2:
        return "晴れ時々くもり"
    if code == 3:
        return "くもり"
    if code in {45, 48}:
        return "霧"
    if code in {51, 53, 55}:
        return "霧雨"
    if code in {56, 57}:
        return "凍結性霧雨"
    if code in {61, 63, 65}:
        return "雨"
    if code in {66, 67}:
        return "凍雨"
    if code in {71, 73, 75, 77}:
        return "雪"
    if code in {80, 81, 82}:
        return "にわか雨"
    if code in {85, 86}:
        return "にわか雪"
    if code in {95, 96, 99}:
        return "雷雨"
    return "不明"


def compass_from_degrees(degrees: float | None, wind_speed_ms: float) -> str:
    if degrees is None:
        return "不明"
    if wind_speed_ms < 0.3:
        return "静穏"
    directions = [
        "北",
        "北北東",
        "北東",
        "東北東",
        "東",
        "東南東",
        "南東",
        "南南東",
        "南",
        "南南西",
        "南西",
        "西南西",
        "西",
        "西北西",
        "北西",
        "北北西",
    ]
    idx = int(((degrees % 360.0) + 11.25) // 22.5) % 16
    return directions[idx]


def fetch_current_weather(session: requests_pkg.Session, lat: float, lon: float) -> CurrentWeatherSnapshot | None:
    url = "https://api.open-meteo.com/v1/jma"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "weather_code,precipitation,wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Tokyo",
        "cell_selection": "land",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }

    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    current = data.get("current") if isinstance(data, dict) else None
    if not isinstance(current, dict):
        return None

    weather_code_raw = current.get("weather_code")
    try:
        weather_code = int(weather_code_raw) if weather_code_raw is not None else None
    except (TypeError, ValueError):
        weather_code = None

    precipitation_raw = current.get("precipitation")
    wind_speed_raw = current.get("wind_speed_10m")
    wind_direction_raw = current.get("wind_direction_10m")

    try:
        precipitation_mm = float(precipitation_raw) if precipitation_raw is not None else 0.0
    except (TypeError, ValueError):
        precipitation_mm = 0.0

    try:
        wind_speed_ms = float(wind_speed_raw) if wind_speed_raw is not None else 0.0
    except (TypeError, ValueError):
        wind_speed_ms = 0.0

    try:
        wind_direction_deg = float(wind_direction_raw) if wind_direction_raw is not None else None
    except (TypeError, ValueError):
        wind_direction_deg = None

    return CurrentWeatherSnapshot(
        weather_label=weather_label_from_code(weather_code),
        precipitation_mm=precipitation_mm,
        wind_direction_label=compass_from_degrees(wind_direction_deg, wind_speed_ms),
        wind_speed_ms=wind_speed_ms,
        updated_at=datetime.now(JST),
    )


def ansi_fg_rgb(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\x1b[38;2;{r};{g};{b}m"


def ansi_bg_rgb(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\x1b[48;2;{r};{g};{b}m"


def mix_rgb(base: Tuple[int, int, int], color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    factor = max(0.0, min(1.0, factor))
    return (
        int(round(base[0] + (color[0] - base[0]) * factor)),
        int(round(base[1] + (color[1] - base[1]) * factor)),
        int(round(base[2] + (color[2] - base[2]) * factor)),
    )


def radar_bg_style_from_pixel(pixel: Tuple[int, int, int, int], use_color: bool) -> str:
    if not use_color:
        return ""

    r, g, b, a = pixel
    if a == 0:
        return ansi_bg_rgb(RADAR_BASE_RGB)

    factor = RADAR_BLEND * (a / 255.0)
    mixed = mix_rgb(RADAR_BASE_RGB, (r, g, b), factor)
    return ansi_bg_rgb(mixed)


def load_radar_target_time(session: requests_pkg.Session) -> Tuple[str, str] | None:
    try:
        resp = session.get(RADAR_TARGET_URL, timeout=60)
        resp.raise_for_status()
        items = resp.json()
        if isinstance(items, list):
            for item in items:
                if "hrpns" in item.get("elements", []):
                    basetime = item.get("basetime")
                    validtime = item.get("validtime")
                    if basetime and validtime:
                        return basetime, validtime
    except Exception:
        pass
    return None


def load_radar_tile(
    session: requests_pkg.Session,
    basetime: str,
    validtime: str,
    zoom: int,
    x: int,
    y: int,
) -> PILImageType | None:
    url = RADAR_TILE_URL.format(
        basetime=basetime,
        validtime=validtime,
        zoom=zoom,
        x=x,
        y=y,
    )
    try:
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        content = resp.content
        if not content:
            return None
        with Image.open(io.BytesIO(content)) as img:
            return img.convert("RGBA")
    except Exception:
        return None


def prefetch_radar_tiles(
    session: requests_pkg.Session,
    basetime: str,
    validtime: str,
    zoom: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> Dict[Tuple[int, int], PILImageType]:
    tiles: Dict[Tuple[int, int], PILImageType] = {}
    for tile_y in range(y_min, y_max + 1):
        for tile_x in range(x_min, x_max + 1):
            tile = load_radar_tile(session, basetime, validtime, zoom, tile_x, tile_y)
            if tile is not None:
                tiles[(tile_x, tile_y)] = tile
    return tiles


def build_radar_background(
    session: requests_pkg.Session,
    sampling: RadarSampling,
    radar_state: RadarState,
    use_color: bool,
) -> Tuple[List[List[str]], str]:
    rows = len(sampling.cells)
    cols = len(sampling.cells[0]) if rows else 0
    backgrounds = [[""] * cols for _ in range(rows)]
    if not use_color:
        return backgrounds, "Radar: disabled"

    target = load_radar_target_time(session)
    if target is None:
        if radar_state.target is not None and radar_state.tiles:
            target = radar_state.target
        else:
            fill = ansi_bg_rgb(RADAR_BASE_RGB)
            for row in backgrounds:
                for idx in range(len(row)):
                    row[idx] = fill
            return backgrounds, "Radar: unavailable"

    if radar_state.target != target or not radar_state.tiles:
        basetime, validtime = target
        x_min, x_max, y_min, y_max = sampling.tile_bounds
        tiles = prefetch_radar_tiles(session, basetime, validtime, RADAR_ZOOM, x_min, x_max, y_min, y_max)
        if tiles:
            radar_state.target = target
            radar_state.tiles = tiles

    if not radar_state.tiles:
        fill = ansi_bg_rgb(RADAR_BASE_RGB)
        for row in backgrounds:
            for idx in range(len(row)):
                row[idx] = fill
        return backgrounds, "Radar: unavailable"

    fill = ansi_bg_rgb(RADAR_BASE_RGB)
    for row_idx in range(rows):
        for col_idx in range(cols):
            sample = sampling.cells[row_idx][col_idx]
            tile = radar_state.tiles.get((sample.tile_x, sample.tile_y))
            if tile is None:
                backgrounds[row_idx][col_idx] = fill
                continue

            pixel = tile.getpixel((sample.px, sample.py))
            if isinstance(pixel, tuple) and len(pixel) == 4:
                rgba = (int(pixel[0]), int(pixel[1]), int(pixel[2]), int(pixel[3]))
            else:
                rgba = (0, 0, 0, 0)
            backgrounds[row_idx][col_idx] = radar_bg_style_from_pixel(rgba, True)

    if radar_state.target is None:
        return backgrounds, "Radar: unavailable"
    basetime, validtime = radar_state.target
    return backgrounds, f"Radar: hrpns {basetime}/{validtime} z{RADAR_ZOOM}"


def extract_visible_tiles(img: PILImageType, cols: int, rows: int, tile_h: int) -> Any:
    arr = np.asarray(img, dtype=np.uint8)
    tiles = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        y0 = r * tile_h
        y1 = y0 + tile_h
        row_slice = arr[y0:y1, :]
        for c in range(cols):
            x0 = c * TILE_W
            x1 = x0 + TILE_W
            tiles[r, c] = row_slice[:, x0:x1]
    return tiles


def resolve_font_path() -> Path | None:
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        "consola.ttf",
        "CascadiaMono.ttf",
        "cascadiamono.ttf",
        "lucon.ttf",
        "cour.ttf",
        "DejaVuSansMono.ttf",
        "dejavusansmono.ttf",
        "NotoSansMono-Regular.ttf",
    ]
    for name in candidates:
        path = font_dir / name
        if path.exists():
            return path

    try:
        from matplotlib import font_manager as mpl_font_manager

        font_path = mpl_font_manager.findfont(
            mpl_font_manager.FontProperties(
                family=["Consolas", "Cascadia Mono", "DejaVu Sans Mono", "Courier New", "monospace"]
            ),
            fallback_to_default=True,
        )
        if font_path:
            path = Path(font_path)
            if path.exists():
                return path
    except Exception:
        pass

    return None


def load_monospace_font() -> PILImageFont_pkg.FreeTypeFont | PILImageFont_pkg.ImageFont:
    font_path = resolve_font_path()
    if font_path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(font_path), size=22)


def build_radar_sampling(
    lat: float,
    lon: float,
    cols: int,
    rows: int,
    scale_m: float,
    ground_aspect: float,
) -> RadarSampling:
    south, west, north, east = viewport_bbox(lat, lon, cols, rows, scale_m, ground_aspect, padding_cells=1.5)
    tile_bounds = webmercator_tile_bounds_from_bbox(south, west, north, east, RADAR_ZOOM)
    cells: List[List[RadarCellSample]] = []
    for row_idx in range(rows):
        row_samples: List[RadarCellSample] = []
        for col_idx in range(cols):
            east_m, north_m = cell_center_meters(col_idx, row_idx, cols, rows, scale_m, ground_aspect)
            cell_lat, cell_lon = meters_to_latlon(east_m, north_m, lat, lon)
            tile_xf, tile_yf = webmercator_tile_xy(cell_lat, cell_lon, RADAR_ZOOM)
            tile_x = int(math.floor(tile_xf))
            tile_y = int(math.floor(tile_yf))
            px = int((tile_xf - tile_x) * 256.0)
            py = int((tile_yf - tile_y) * 256.0)
            row_samples.append(
                RadarCellSample(
                    tile_x=tile_x,
                    tile_y=tile_y,
                    px=max(0, min(255, px)),
                    py=max(0, min(255, py)),
                )
            )
        cells.append(row_samples)
    return RadarSampling(cells=cells, tile_bounds=tile_bounds)


def build_glyph_atlas(font: Any, tile_h: int) -> Tuple[List[str], Any]:
    box_drawing = list("─│┌┐└┘├┤┬┴┼╭╮╯╰╱╲╳")
    ascii_line = list("-|/\\+=_*^~.:")
    digits = list(string.digits)
    uppercase = list("AEFHIKLMNOSTUVWXYZ")
    lowercase = list("aefhiklmnostuvwxyz")
    glyphs = list(dict.fromkeys(box_drawing + ascii_line + digits + uppercase + lowercase))

    atlas_images = []
    atlas_chars = []
    for ch in glyphs:
        img = Image.new("L", (TILE_W, tile_h), 0)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), ch, font=font)
        if bbox is None:
            continue
        left, top, right, bottom = bbox
        text_w = right - left
        text_h = bottom - top
        x = (TILE_W - text_w) / 2 - left
        y = (tile_h - text_h) / 2 - top
        draw.text((x, y), ch, fill=255, font=font)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.max() == 0:
            continue
        atlas_chars.append(ch)
        atlas_images.append(arr)

    if not atlas_images:
        raise RuntimeError("No glyph atlas could be built.")

    return atlas_chars, np.stack(atlas_images, axis=0)


def glyph_bias(ch: str) -> float:
    if ch in "─│┌┐└┘├┤┬┴┼╭╮╯╰╱╲╳":
        return 0.0
    if ch in "-|/\\+=_*^~.:":
        return 0.02
    if ch.isdigit():
        return 0.05
    return 0.12


def choose_glyph_for_tile(
    tile: Any,
    atlas_chars: List[str],
    atlas_stack: Any,
    cache: Dict[bytes, str],
) -> str:
    if tile.max() == 0:
        return " "

    key = np.ascontiguousarray(tile).tobytes()
    cached = cache.get(key)
    if cached is not None:
        return cached

    target = tile.astype(np.float32) / 255.0
    diff = atlas_stack - target[None, :, :]
    scores = np.mean(diff * diff, axis=(1, 2))
    # scores = scores + np.array([glyph_bias(ch) for ch in atlas_chars], dtype=np.float32)
    glyph = atlas_chars[int(np.argmin(scores))]
    cache[key] = glyph
    return glyph


def render_layer(
    img: PILImageType,
    atlas_chars: List[str],
    atlas_stack: Any,
    cols: int,
    rows: int,
    tile_h: int,
) -> List[List[str]]:
    tiles = extract_visible_tiles(img, cols, rows, tile_h)
    cache: Dict[bytes, str] = {}
    grid = [[" "] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            grid[r][c] = choose_glyph_for_tile(tiles[r, c], atlas_chars, atlas_stack, cache)
    return grid


def compose_layers(
    layers: List[Tuple[LayerSpec, List[List[str]]]],
    cols: int,
    rows: int,
    use_color: bool,
) -> Tuple[List[List[str]], List[List[str]]]:
    chars = [[" "] * cols for _ in range(rows)]
    styles = [[""] * cols for _ in range(rows)]
    ranks = [[10_000] * cols for _ in range(rows)]

    for layer, grid in layers:
        for r in range(rows):
            for c in range(cols):
                glyph = grid[r][c]
                if glyph == " ":
                    continue
                if layer.priority <= ranks[r][c]:
                    chars[r][c] = glyph
                    styles[r][c] = layer.style if use_color else ""
                    ranks[r][c] = layer.priority

    return chars, styles


def format_layers(
    chars: List[List[str]],
    styles: List[List[str]],
    backgrounds: List[List[str]],
    use_color: bool,
) -> str:
    lines = []
    for r in range(len(chars)):
        pieces = []
        last_fg = None
        last_bg = None
        for c in range(len(chars[r])):
            fg_style = styles[r][c]
            bg_style = backgrounds[r][c] if r < len(backgrounds) and c < len(backgrounds[r]) else ""
            glyph = chars[r][c]
            if use_color:
                if bg_style != last_bg:
                    pieces.append(bg_style if bg_style else RESET_BG_STYLE)
                    last_bg = bg_style
                if fg_style != last_fg:
                    pieces.append(fg_style if fg_style else RESET_FG_STYLE)
                    last_fg = fg_style
            pieces.append(glyph)
        if use_color:
            pieces.append(RESET_STYLE)
        lines.append("".join(pieces))
    return "\n".join(lines)


def render_scene(
    scene: StaticScene,
    radar_backgrounds: List[List[str]],
    weather: CurrentWeatherSnapshot,
    use_color: bool,
) -> str:
    prefix = f"{MARKER_STYLE}+{RESET_STYLE} " if use_color else "+ "
    header = (
        f"{prefix}現在地：{scene.location_label} | {weather.weather_label} | "
        f"降水量：{weather.precipitation_mm:.1f}mm | 風：{weather.wind_direction_label}{weather.wind_speed_ms:.1f}m/s | "
        f"最終更新：{format_jst_timestamp(weather.updated_at)}"
    )
    rendered = format_layers(scene.chars, scene.styles, radar_backgrounds, use_color)
    return "\n".join([header, rendered])


def apply_marker_to_grid(
    chars: List[List[str]],
    styles: List[List[str]],
    lat: float,
    lon: float,
    center_lat: float,
    center_lon: float,
    cols: int,
    rows: int,
    scale_m: float,
    ground_aspect: float,
    use_color: bool,
) -> None:
    x, y = project_to_meters(lat, lon, center_lat, center_lon)
    half_w = cols * scale_m / 2.0
    half_h = rows * scale_m * ground_aspect / 2.0
    if x < -half_w or x > half_w or y < -half_h or y > half_h:
        return

    col = int(math.floor((x + half_w) / scale_m))
    row = int(math.floor((half_h - y) / (scale_m * ground_aspect)))
    col = max(0, min(cols - 1, col))
    row = max(0, min(rows - 1, row))

    if row < len(chars) and col < len(chars[row]):
        chars[row][col] = "+"
        styles[row][col] = MARKER_STYLE if use_color else ""


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except ValueError:
                pass

    parser = argparse.ArgumentParser(
        description="Render an ASCII map with administrative boundaries and JMA rain radar overlay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Map center latitude.")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Map center longitude.")
    parser.add_argument(
        "--marker-lat",
        type=float,
        default=DEFAULT_MARKER_LAT,
        help="Marker latitude.",
    )
    parser.add_argument(
        "--marker-lon",
        type=float,
        default=DEFAULT_MARKER_LON,
        help="Marker longitude.",
    )
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE_METERS, help="Meters per character cell.")
    parser.add_argument("--cols", type=int, default=None, help="Map width in characters.")
    parser.add_argument("--rows", type=int, default=None, help="Map height in characters.")
    parser.add_argument(
        "--cell-aspect",
        type=float,
        default=DEFAULT_CELL_ASPECT,
        help="Terminal cell height/width ratio.",
    )
    parser.add_argument(
        "--auto-cell-aspect",
        action="store_true",
        help="Detect the terminal cell height/width ratio instead of using the fixed default.",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=300.0,
        help="Seconds between radar refreshes; set 0 for a single render.",
    )
    parser.add_argument("--once", action="store_true", help="Render once and exit.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    args = parser.parse_args()

    load_runtime_dependencies()

    term = get_terminal_size(fallback=(100, 40))
    cols = args.cols or max(20, min(120, term.columns - 1))
    rows = args.rows or max(10, min(60, term.lines - 4))
    use_color = not args.no_color and os.environ.get("NO_COLOR") is None and sys.stdout.isatty()
    terminal_aspect = detect_terminal_cell_aspect() if args.auto_cell_aspect else args.cell_aspect
    ground_aspect = terminal_aspect
    glyph_scale = ground_aspect / (TILE_H / TILE_W)
    glyph_tile_h = max(4, int(round(TILE_H * glyph_scale)))
    live_refresh = not args.once and args.refresh_seconds > 0 and sys.stdout.isatty()

    base_dir = Path(__file__).resolve().parent
    cache_dir = ensure_cache_dir(base_dir)
    session = build_session()
    font = load_monospace_font()
    atlas_chars, atlas_stack = build_glyph_atlas(font, glyph_tile_h)
    radar_sampling = build_radar_sampling(args.lat, args.lon, cols, rows, args.scale, ground_aspect)

    reverse = session.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={
            "format": "jsonv2",
            "lat": args.lat,
            "lon": args.lon,
            "zoom": 18,
            "addressdetails": 1,
        },
        timeout=60,
    )
    reverse.raise_for_status()
    address = reverse.json().get("address", {})

    pref_name = resolve_address_component(address, ["state", "province"], "unknown")
    city_name = resolve_address_component(address, ["city", "town", "village", "municipality"], "unknown")
    ward_name = resolve_address_component(
        address,
        ["city_district", "borough", "district", "suburb", "quarter", "neighbourhood"],
        "unknown",
    )
    location_label = format_location_label(pref_name, city_name, ward_name)

    elements = fetch_admin_boundaries_in_bbox(
        session,
        cache_dir,
        args.lat,
        args.lon,
        cols,
        rows,
        args.scale,
        ground_aspect,
    )
    features = parse_boundary_features(elements)

    grouped: Dict[Tuple[str, int], List[BoundaryFeature]] = {}
    for feature in features:
        grouped.setdefault((feature.style, feature.priority), []).append(feature)

    layers: List[Tuple[LayerSpec, List[List[str]]]] = []
    for (style, priority), items in sorted(grouped.items(), key=lambda item: item[0][1]):
        canvas = rasterize_features_to_canvas(
            items,
            args.lat,
            args.lon,
            cols,
            rows,
            args.scale,
            glyph_tile_h,
            ground_aspect,
        )
        grid = render_layer(canvas, atlas_chars, atlas_stack, cols, rows, glyph_tile_h)
        name = f"admin-level-{priority}"
        layers.append((LayerSpec(name, style, priority), grid))

    chars, styles = compose_layers(layers, cols, rows, use_color)
    apply_marker_to_grid(
        chars,
        styles,
        args.marker_lat,
        args.marker_lon,
        args.lat,
        args.lon,
        cols,
        rows,
        args.scale,
        ground_aspect,
        use_color,
    )

    scene = StaticScene(
        chars=chars,
        styles=styles,
        location_label=location_label,
        radar_sampling=radar_sampling,
    )
    radar_state = RadarState()

    def emit_frame() -> None:
        current_weather = fetch_current_weather(session, args.lat, args.lon)
        if current_weather is None:
            current_weather = CurrentWeatherSnapshot(
                weather_label="不明",
                precipitation_mm=0.0,
                wind_direction_label="不明",
                wind_speed_ms=0.0,
                updated_at=datetime.now(JST),
            )

        radar_backgrounds, _ = build_radar_background(
            session,
            scene.radar_sampling,
            radar_state,
            use_color and CENTER_RAIN_RADAR,
        )
        frame = render_scene(scene, radar_backgrounds, current_weather, use_color)
        if live_refresh:
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(frame)
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            print(frame)

    if live_refresh:
        try:
            while True:
                emit_frame()
                time.sleep(args.refresh_seconds)
        except KeyboardInterrupt:
            return 130
    else:
        emit_frame()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
