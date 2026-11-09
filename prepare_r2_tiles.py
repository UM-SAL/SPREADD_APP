#!/usr/bin/env python
"""prepare_r2_tiles.py — Run ONCE locally (needs rasterio; use the `geo_env` conda env).

Pre-tiles every raster onto the HII master ~98 m grid (the same reference grid
`rebekahs_model.compute_per_pixel_total` uses), writing 512x512 zlib-compressed
float32 tiles + a catalog.json. Upload the contents of OUT_DIR to Cloudflare R2.

Differences from the original one-shot script:
  * Reprojects each raster ONCE onto the full master grid, then slices tiles
    (the original issued one warp per tile -> thousands of warps). Nearest
    resampling makes the sliced result identical to the per-tile version.
  * Adds crs_wkt / crs_epsg_hint to the catalog because the source rasters carry
    a LOCAL_CS["NAD83 / Conus Albers"] tag (really EPSG:5070) that bare readers
    can't reproject; the hint lets the read-side force 5070.

Run:  conda activate geo_env && python prepare_r2_tiles.py
(The geo_env activation sets PROJ_DATA so transform_bounds works.)
"""
import json
import zlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window

RASTER_DIR = Path("/Users/rustin/Documents/Spatial_Analysis_Lab/HF_Datasets/Full_Rasters/All_Rasters")
OUT_DIR = Path("/Users/rustin/Documents/Spatial_Analysis_Lab/HF_Datasets/Full_Rasters/Rasters_Tiled")
TILE_SIZE = 512
CRS_EPSG_HINT = 5070  # source rasters are Conus Albers tagged as a bare LOCAL_CS


def build_tiles():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tifs = sorted(set(RASTER_DIR.glob("**/*.tif")))
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in {RASTER_DIR}")

    # Use HII.tif as the master ~98m reference grid (same as compute_per_pixel_total).
    ref_tif = next((p for p in tifs if p.stem.lower() == "hii"), tifs[0])
    with rasterio.open(ref_tif) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        height, width = ref.height, ref.width
        bounds_4326 = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds, densify_pts=21)

    catalog = {
        "crs": dst_crs.to_string(),
        "crs_wkt": dst_crs.to_wkt(),
        "crs_epsg_hint": CRS_EPSG_HINT,
        "transform": [dst_transform.a, dst_transform.b, dst_transform.c,
                      dst_transform.d, dst_transform.e, dst_transform.f],
        "width": width,
        "height": height,
        "tile_size": TILE_SIZE,
        "bounds_4326": list(bounds_4326),
        "stems": [p.stem for p in tifs],
    }

    summary = {}
    for tif_path in tifs:
        stem = tif_path.stem
        stem_dir = OUT_DIR / stem
        stem_dir.mkdir(exist_ok=True)

        with rasterio.open(tif_path) as src:
            same_grid = (src.crs == dst_crs and src.transform == dst_transform
                         and src.width == width and src.height == height)
            nodata = src.nodata

            # Build the full raster on the master grid once (float32, NaN outside).
            grid = np.full((height, width), np.nan, dtype=np.float32)
            if same_grid:
                raw = src.read(1).astype(np.float32)
                if nodata is not None:
                    raw = np.where(raw == nodata, np.nan, raw)
                grid[:] = raw
            else:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=grid,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    src_nodata=nodata,
                    dst_nodata=np.nan,
                )

        saved = 0
        for row_off in range(0, height, TILE_SIZE):
            for col_off in range(0, width, TILE_SIZE):
                win_h = min(TILE_SIZE, height - row_off)
                win_w = min(TILE_SIZE, width - col_off)
                tile_arr = grid[row_off:row_off + win_h, col_off:col_off + win_w]

                # Only save non-empty tiles to save space.
                if not np.isnan(tile_arr).all():
                    tx = col_off // TILE_SIZE
                    ty = row_off // TILE_SIZE
                    compressed = zlib.compress(np.ascontiguousarray(tile_arr).tobytes(), level=6)
                    (stem_dir / f"{ty}_{tx}.bin").write_bytes(compressed)
                    saved += 1
        summary[stem] = saved
        print(f"  {stem:20} {saved} tiles ({'same-grid' if same_grid else 'reprojected'})")

    (OUT_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2))
    total = sum(summary.values())
    print(f"\nDone. {total} tiles across {len(tifs)} layers -> {OUT_DIR}")
    print("Upload the contents of that folder to your Cloudflare R2 bucket.")


if __name__ == "__main__":
    build_tiles()
