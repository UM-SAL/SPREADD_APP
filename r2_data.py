"""Cloudflare R2 raster access for the browser (stlite / Pyodide) build of S.P.R.E.A.D.D.

Pyodide cannot install rasterio/rasterstats, so the 17 model rasters were pre-aligned
onto one ~98 m grid (EPSG:5070) and sliced into zlib-compressed float32 512x512 tiles
by ``prepare_r2_tiles.py`` and uploaded to a public Cloudflare R2 bucket. This module
fetches only the tiles overlapping a polygon and assembles plain numpy arrays, so the
rest of the model runs in pure numpy + shapely.

Nothing here needs rasterio, rasterstats or huggingface_hub, and no credentials: the
bucket is read with its public r2.dev URL.
"""
from __future__ import annotations

import json
import zlib
import urllib.request
from functools import lru_cache

import numpy as np
from shapely import contains_xy

# Public Cloudflare R2 bucket (read-only, no credentials). Contents of r2_upload/
# live at the bucket root: catalog.json and <stem>/<ty>_<tx>.bin tiles.
R2_BASE_URL = "https://pub-b282d5bd73c146caa09344d1698a7081.r2.dev"


@lru_cache(maxsize=4096)
def fetch_r2_bytes(rel_path: str) -> bytes | None:
    """Fetch a file/tile from R2. Works in stlite (sync XHR) and desktop Python (urllib).

    Cached so a tile shared by several polygons/layers in one run is fetched once.
    Returns None if the object is missing (e.g. an all-NaN tile that was never written).
    """
    url = f"{R2_BASE_URL.rstrip('/')}/{rel_path.lstrip('/')}"
    # Pyodide / stlite web worker: use a synchronous XMLHttpRequest.
    try:
        from js import XMLHttpRequest  # type: ignore

        xhr = XMLHttpRequest.new()
        xhr.open("GET", url, False)
        xhr.responseType = "arraybuffer"
        xhr.send(None)
        if xhr.status == 200:
            return bytes(xhr.response.to_py())
        return None
    except ImportError:
        # Standard CPython (desktop Streamlit / parity check). r2.dev 403s the default
        # urllib User-Agent, so send a browser-like one.
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SPREADD)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception:
            return None


@lru_cache(maxsize=1)
def get_r2_catalog() -> dict:
    """Load and cache catalog.json (grid CRS, affine transform, size, tile layout, stems)."""
    raw = fetch_r2_bytes("catalog.json")
    if not raw:
        raise RuntimeError(
            f"Could not load catalog.json from Cloudflare R2 ({R2_BASE_URL}). "
            "Check the bucket's public access is enabled and CORS allows GET."
        )
    return json.loads(raw.decode("utf-8"))


def catalog_stems() -> list[str]:
    """Ordered list of raster stems available in the bucket (e.g. 'HII', 'Spotted_Knap')."""
    return list(get_r2_catalog()["stems"])


def read_window_from_r2(stem: str, minx: float, miny: float, maxx: float, maxy: float,
                        catalog: dict | None = None):
    """Assemble a float32 window of ``stem`` covering (minx, miny, maxx, maxy) in grid CRS.

    Fetches only the 512x512 tiles overlapping the bounds. Missing tiles stay NaN.
    Returns (array, win_transform) where win_transform = (res_x, 0, minx, 0, -res_y, maxy)
    describes the returned array's georeferencing on the global grid.
    """
    if catalog is None:
        catalog = get_r2_catalog()
    a, b, c, d, e, f = catalog["transform"]  # a=res_x, e=-res_y, c=origin_x, f=origin_y
    res_x, res_y = abs(a), abs(e)
    width, height = catalog["width"], catalog["height"]
    ts = catalog["tile_size"]

    # Column/row range on the global grid (pad 2 px, clamp to the raster extent).
    col0 = max(0, int(np.floor((minx - c) / res_x)) - 2)
    col1 = min(width, int(np.ceil((maxx - c) / res_x)) + 2)
    row0 = max(0, int(np.floor((f - maxy) / res_y)) - 2)
    row1 = min(height, int(np.ceil((f - miny) / res_y)) + 2)
    col1 = max(col1, col0 + 1)
    row1 = max(row1, row0 + 1)

    out_h, out_w = row1 - row0, col1 - col0
    out_arr = np.full((out_h, out_w), np.nan, dtype=np.float32)

    tx0, tx1 = col0 // ts, (col1 - 1) // ts
    ty0, ty1 = row0 // ts, (row1 - 1) // ts

    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            raw = fetch_r2_bytes(f"{stem}/{ty}_{tx}.bin")
            if not raw:
                continue  # tile was all-NaN and never written; leave NaN
            t_row0, t_col0 = ty * ts, tx * ts
            t_h = min(ts, height - t_row0)
            t_w = min(ts, width - t_col0)
            tile = np.frombuffer(zlib.decompress(raw), dtype=np.float32).reshape((t_h, t_w))

            r_start = max(row0, t_row0)
            r_end = min(row1, t_row0 + t_h)
            c_start = max(col0, t_col0)
            c_end = min(col1, t_col0 + t_w)
            if r_end <= r_start or c_end <= c_start:
                continue
            out_arr[r_start - row0:r_end - row0, c_start - col0:c_end - col0] = (
                tile[r_start - t_row0:r_end - t_row0, c_start - t_col0:c_end - t_col0]
            )

    win_minx = c + col0 * res_x
    win_maxy = f - row0 * res_y
    win_transform = (res_x, 0.0, win_minx, 0.0, -res_y, win_maxy)
    return out_arr, win_transform


def polygon_mask_on_window(geom_proj, out_shape, win_transform, all_touched: bool = False):
    """Boolean mask (True inside polygon) for a window, via Shapely 2.0 contains_xy.

    ``geom_proj`` must already be in the grid CRS. Pixel centers inside the polygon are
    True (matches rasterstats' default all_touched=False). When all_touched=True the
    center mask is dilated by one pixel as a pure-numpy approximation of GDAL's
    any-touched rasterization (good enough for the optional toggle).
    """
    res_x, _, minx, _, neg_res_y, maxy = win_transform
    res_y = abs(neg_res_y)
    h, w = out_shape
    xs = minx + (np.arange(w) + 0.5) * res_x
    ys = maxy - (np.arange(h) + 0.5) * res_y
    xx, yy = np.meshgrid(xs, ys)
    mask = np.asarray(contains_xy(geom_proj, xx, yy), dtype=bool)
    if all_touched and mask.any():
        m = mask.copy()
        m[:-1, :] |= mask[1:, :]
        m[1:, :] |= mask[:-1, :]
        m[:, :-1] |= mask[:, 1:]
        m[:, 1:] |= mask[:, :-1]
        mask = m
    return mask
