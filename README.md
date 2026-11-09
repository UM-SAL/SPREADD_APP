# S.P.R.E.A.D.D. — browser build

**Spatial Probability of Risk from Environmental Action Driven Dispersal** — a fully
in-browser version of the UM Spatial Analysis Lab invasive-species risk tool, built to run
on **GitHub Pages** with [stlite](https://github.com/whitphx/stlite) (Streamlit compiled to
WebAssembly via Pyodide). No server, no Python install for the user.

## How it works

Pyodide can't install `rasterio`/`rasterstats`, so the 17 model rasters were pre-aligned
onto one ~98 m grid (EPSG:5070) and sliced into zlib-compressed float32 512×512 tiles
(`prepare_r2_tiles.py`) and uploaded to a public **Cloudflare R2** bucket. When a user runs
a prediction, their browser fetches only the handful of tiles overlapping their polygon and
computes the zonal-statistics table and the per-pixel threat maps in pure `numpy` + `shapely`.

| File | Role |
|------|------|
| `index.html` | Mounts the app in stlite and lists the Pyodide requirements |
| `app.py` | Streamlit UI (draw/upload a polygon → risk table + per-pixel maps) |
| `rebekahs_model.py` | A+B+C+D decision model; R2-based zonal stats + per-pixel grids |
| `r2_data.py` | Fetches R2 catalog/tiles and assembles numpy windows + polygon masks |
| `Species_List 1(*).csv` | Management-action, final-output and citation lookup tables |
| `SPREADD_diagram_v2.png` | Model diagram shown under "How the model works" |

The public raster bucket URL is set in `r2_data.py` (`R2_BASE_URL`). No credentials are used
or needed — the bucket is read-only public with CORS allowing GET.

## Deploy (GitHub Pages)

Push to the default branch, then **Settings → Pages → Deploy from a branch → `main` / root**.
GitHub serves the static app at `https://<org>.github.io/<repo>/`.

## Local testing (optional, desktop Streamlit)

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

This hits the same Cloudflare R2 bucket over HTTPS, so it behaves like the browser build.

## Regenerating the tiles

Run `prepare_r2_tiles.py` (needs `rasterio`, run outside the browser) against the source
`Full_Rasters/*.tif`, then upload the `r2_upload/` output to the R2 bucket root with
`rclone copy r2_upload/ r2:spreadd-rasters/`.
