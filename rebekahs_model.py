from __future__ import annotations  # postpone evaluation of annotations for forward refs

import re  # regex for safe column name sanitization
import math  # finiteness checks
from pathlib import Path  # path handling
from typing import Optional, Sequence  # typing helpers

import numpy as np  # numeric ops (variance, NaN handling)
import pandas as pd  # tabular wrapper for zonal stats output
import geopandas as gpd  # vector GIS I/O and CRS handling
from shapely.geometry import box  # raster footprint checks

# Browser build: rasters come from Cloudflare R2 tiles (no rasterio/rasterstats in Pyodide).
from r2_data import (
    get_r2_catalog,
    catalog_stems,
    read_window_from_r2,
    polygon_mask_on_window,
)


def safe_colname(name_seed: str, suffix: str, existing: set[str], max_len: int = 63) -> str:  # sanitize/unique colnames
    """  # docstring start
    Make a safe column name. Default max_len=63 (fine for GeoPackage/Parquet).  # docstring line
    If you need shapefile-compatible, set max_len=10.  # docstring line
    """  # docstring end
    base = re.sub(r"[^A-Za-z0-9]", "", str(name_seed)).upper() or "RAST"  # strip non-alnum, uppercase, default
    suf = re.sub(r"[^A-Za-z0-9_]", "", str(suffix)).upper()  # keep alnum/underscore in suffix
    if not suf.startswith("_"):  # enforce underscore prefix
        suf = "_" + suf  # add underscore

    max_base = max_len - len(suf)  # how many chars left for base
    if max_base < 1:  # suffix too long to fit
        raise ValueError(f"Suffix '{suf}' too long for max_len={max_len}.")  # hard error

    cand = f"{base[:max_base]}{suf}"  # candidate name trimmed to length
    if cand not in existing:  # unique on first try
        existing.add(cand)  # track used name
        return cand  # return first candidate

    for i in range(1, 1000):  # try numeric disambiguation
        extra = str(i)  # numeric suffix
        max_base2 = max_len - len(suf) - len(extra)  # leave space for number
        if max_base2 < 1:  # can't fit this number
            continue  # try next
        cand = f"{base[:max_base2]}{suf}{extra}"  # candidate with numeric suffix
        if cand not in existing:  # unique
            existing.add(cand)  # track used
            return cand  # return unique name

    raise RuntimeError(f"Could not create unique colname for {name_seed}{suffix}.")  # give up after 999


def iter_rasters(raster_dir: Path, patterns: Optional[Sequence[str]] = None) -> list[Path]:  # list rasters in dir
    """  # docstring start
    List rasters in a directory. Patterns default to common raster extensions.  # docstring line
    """  # docstring end
    if patterns is None:  # default patterns when not provided
        patterns = ["*.tif", "*.tiff", "*.img", "*.bil"]  # common raster extensions
    rasters: list[Path] = []  # accumulator list
    for pat in patterns:  # iterate patterns
        rasters.extend(raster_dir.glob(pat))  # add matches for each pattern
    return sorted(set(rasters))  # de-duplicate and sort


# -------------------- Layer scoring (replaces the old median_divisor) --------------------
# The decision tree adds four 0-1 inputs (A + B + C + D). The old code divided every raster
# by one constant (median_divisor), but the rasters are in different units and one points the
# other way, so each layer is now converted to a 0-1 score using its own units instead.
# Cells equal to a layer's fill value are treated as no data and never enter the statistics.
#
#   Layer              Raw values                                  0-1 score                   Fill (no data)
#   species rasters    INHABIT habitat suitability, 0-100 %        value / 100                 0 (outside model), 127 (water/no data)
#   HII                recreated Human Influence Index, 0,10..50   value / 50                  none (0 = no mapped influence)
#   Biodiversity       Biodiversity Intactness, 0-100 % retained   1 - value / 100 (% lost)    0 (outside coverage)
#   Clim_Velocity      climate velocity, km/yr                     min(value / 7, 1)           0 (outside coverage)
#
# 7 km/yr is about the 95th percentile of the non-fill Clim_Velocity cells in MT/ID/WY (from the
# raster's histogram), so the fastest-changing ~5% of the region gets the maximum score.

HII_MAX_CLASS = 50.0
CLIM_VELOCITY_CAP_KM_YR = 7.0
SPECIES_FILL_VALUES = (0, 127)

LAYER_SCORING = {
    "hii": {"fill": (), "score": lambda v: np.clip(v / HII_MAX_CLASS, 0.0, 1.0)},
    "biodiversity": {"fill": (0,), "score": lambda v: np.clip(1.0 - v / 100.0, 0.0, 1.0)},
    "climvelocity": {"fill": (0,), "score": lambda v: np.clip(v / CLIM_VELOCITY_CAP_KM_YR, 0.0, 1.0)},
}
SPECIES_SCORING = {"fill": SPECIES_FILL_VALUES, "score": lambda v: np.clip(v / 100.0, 0.0, 1.0)}


def layer_scoring(raster_stem: str) -> dict:
    """Return the fill values and 0-1 scoring function for a raster, by file stem."""
    key = re.sub(r"[^a-z0-9]", "", str(raster_stem).lower())
    return LAYER_SCORING.get(key, SPECIES_SCORING)


def _window_stats_for_stem(stem: str, mask, win_arr):
    """Raw + 0-1 score stats for one stem over a boolean polygon mask and its window.

    Fill values (LAYER_SCORING / SPECIES_FILL_VALUES) and NaN (outside data) are dropped
    before any statistic, matching the old rasterstats zone_func/add_stats behaviour.
    """
    spec = layer_scoring(stem)
    fill = np.asarray(spec["fill"], dtype="float32")
    vals = win_arr[mask]
    good = ~np.isnan(vals)
    if fill.size:
        good &= ~np.isin(vals, fill)
    vals = vals[good].astype(float)
    if vals.size == 0:
        return {"mean": None, "std": None, "median": None, "count": 0,
                "score_median": None, "score_std": None}
    score = np.asarray(spec["score"](vals), dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "count": int(vals.size),
        "score_median": float(np.median(score)),
        "score_std": float(np.std(score)),
    }


def zonal_stats_gdf(zones_gdf, all_touched: bool = False, col_max_len: int = 63):
    """Zonal statistics for every R2 raster stem, per zone polygon — pure numpy, no rasterio.

    Returns a GeoDataFrame (one row per input polygon) with the SAME columns the old
    rasterio/rasterstats path produced, so the rest of the app is unchanged:
      <STEM>_MD   median 0-1 score inside the polygon (the value the decision tree uses)
      <STEM>_VAR  variance of the 0-1 score inside the polygon
      <STEM>_MEAN / _STD / _MED   raw-unit mean, std and median (fill values excluded)
      <STEM>_N    number of valid cells used

    All rasters share one pre-aligned grid (catalog CRS), so each polygon is reprojected
    once and a single pixel mask is reused across all 17 stems.
    """
    catalog = get_r2_catalog()
    grid_crs = catalog["crs"]
    stems = catalog_stems()

    if "geometry" not in zones_gdf.columns:
        raise ValueError("Zones GeoDataFrame has no geometry column.")
    if zones_gdf.crs is None:
        raise ValueError("Zones GeoDataFrame has no CRS. Set zones.crs before running zonal stats.")

    # Geometry cleanup (mirror the old path: make_valid, else buffer(0)).
    zc = zones_gdf.loc[zones_gdf.geometry.notna() & ~zones_gdf.geometry.is_empty].copy()
    try:
        zc["geometry"] = zc["geometry"].make_valid()
    except Exception:
        zc["geometry"] = zc["geometry"].buffer(0)
    zc = zc.loc[zc.geometry.notna() & ~zc.geometry.is_empty].copy()
    if zc.empty:
        raise ValueError("All zones geometries are empty or invalid after cleanup.")

    out = zc.copy()  # keeps zone id + geometry; stats columns are appended below
    zones_grid = zc.to_crs(grid_crs)  # reproject once to the tile grid CRS (EPSG:5070)

    # Reserve a stable, unique column name per stem/suffix (distinct prefixes => no clashes).
    existing_cols = {c.upper() for c in out.columns}
    colnames: dict[str, dict[str, str]] = {}
    for stem in stems:
        names = {suf: safe_colname(stem, suf, existing_cols, max_len=col_max_len)
                 for suf in ("MD", "VAR", "MEAN", "STD", "MED", "N")}
        colnames[stem] = names
        for suf in ("MD", "VAR", "MEAN", "STD", "MED"):
            out[names[suf]] = np.nan
        out[names["N"]] = 0  # count defaults to 0, matching the old fillna(0)

    geoms = zones_grid.geometry.to_numpy()
    for idx, geom in zip(out.index, geoms):
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        if not np.isfinite([minx, miny, maxx, maxy]).all():
            continue
        mask = None  # computed once; identical window/grid for every stem
        for stem in stems:
            arr, win_tr = read_window_from_r2(stem, minx, miny, maxx, maxy, catalog)
            if mask is None:
                mask = polygon_mask_on_window(geom, arr.shape, win_tr, all_touched=all_touched)
            s = _window_stats_for_stem(stem, mask, arr)
            c = colnames[stem]
            out.at[idx, c["MD"]] = s["score_median"]
            out.at[idx, c["VAR"]] = (s["score_std"] ** 2) if s["score_std"] is not None else np.nan
            out.at[idx, c["MEAN"]] = s["mean"]
            out.at[idx, c["STD"]] = s["std"]
            out.at[idx, c["MED"]] = s["median"]
            out.at[idx, c["N"]] = s["count"]

    return out


# -------------------- Decision tree helpers --------------------  # section divider

SPECIES_META_COLS = {"FS-aligned Management Action", "Notes", "Core Disturbance"}  # non-species columns

SPECIES_RASTER_ALIASES = {  # map canonical species -> raster stem aliases
    "spottedknapweed": "spottedknap",  # shorten knapweed raster stem
    "dalmatiantoadflax": "dalmationtoad",  # match raster spelling
    "sulfurcinquefoil": "sulfurcinq",  # shortened cinquefoil
    "rushskeletonweed": "rushskel",  # shortened skeletonweed
    "oxeyedaisy": "oxeye",  # shortened oxeye daisy
    "commonbuckthorn": "buckthorn",  # shortened buckthorn
    "commontansy": "tansy",  # shortened tansy
}  # end alias map

SPECIES_NAME_ALIASES = {  # normalize known name variations across tables
    "dalmationtoadflax": "dalmatiantoadflax",  # fix spelling in Final Output
}  # end name aliases

RESISTANCE_RASTER_KEYS = {  # rasters used for D (resistance)
    "hii": "HII",  # Human Influence Index raster stem
    "biodiversity": "Biodiversity",  # Biodiversity raster stem
    "clim_velocity": "Clim_Velocity",  # Climate Velocity raster stem
}  # end resistance keys


def load_csv_latin1(path: str) -> pd.DataFrame:  # load CSV with permissive encoding
    """Load a CSV using latin-1 encoding (handles odd characters)."""  # docstring
    return pd.read_csv(path, encoding="latin-1")  # read CSV


def _normalize_action_text(text: str) -> str:  # normalize action strings for matching
    """Normalize management action text for matching."""  # docstring
    return str(text).replace("\x96", "-").replace("\u2013", "-").strip().lower()  # normalize dashes and case


def get_treatment_options(species_table: pd.DataFrame) -> list[str]:  # build treatment list from table
    """Return ordered management actions from the species table."""  # docstring
    actions = species_table["FS-aligned Management Action"].dropna().astype(str)  # extract actions
    seen = set()  # track unique items
    ordered = []  # preserve order
    for val in actions:  # iterate in file order
        norm = _normalize_action_text(val)  # normalize for uniqueness
        if norm in seen:  # skip duplicates
            continue  # skip
        seen.add(norm)  # mark seen
        ordered.append(val)  # keep original casing
    return ordered  # return list


def _action_row_for_treatment(species_table: pd.DataFrame, treatment: str) -> pd.Series | None:  # lookup row
    """Find the lookup row that corresponds to the selected treatment."""  # docstring
    actions = species_table["FS-aligned Management Action"].astype(str)  # action column
    actions_norm = actions.apply(_normalize_action_text)  # normalized actions
    treat_norm = _normalize_action_text(treatment)  # normalized treatment
    mask = actions_norm == treat_norm  # exact match mask
    matches = species_table.loc[mask]  # matching rows
    if matches.empty:  # no match
        return None  # return None
    return matches.iloc[0]  # first match


def _column_letter(index: int) -> str:  # convert 0-based index to Excel-like letters
    """Convert a 0-based column index to Excel column letters."""  # docstring
    letters = ""  # output accumulator
    i = index  # working index
    while i >= 0:  # loop until done
        i, rem = divmod(i, 26)  # quotient and remainder
        letters = chr(65 + rem) + letters  # prepend letter
        i -= 1  # shift for Excel-style lettering
    return letters  # return result


def get_final_output_column_options(final_output_df: pd.DataFrame) -> list[tuple[str, str]]:  # build C options
    """Build labeled options for Final Output columns (skip column A and H)."""  # docstring
    options: list[tuple[str, str]] = []  # label/column pairs
    for idx, col in enumerate(final_output_df.columns):  # iterate columns with index
        letter = _column_letter(idx)  # Excel letter
        if letter in ("A", "H"):  # skip Species (A) and H (empty)
            continue  # skip
        if col == "Species":  # extra guard
            continue  # skip
        label = f"Column {letter} ({col})"  # build UI label
        options.append((label, col))  # add pair
    return options  # return list


def _canonicalize(name: str) -> str:  # normalize names for matching
    """Lowercase and remove non-alphanumerics for loose matching."""  # docstring
    return re.sub(r"[^a-z0-9]", "", str(name).lower())  # normalize


def _canonical_species_name(name: str) -> str:  # normalize species names across sources
    """Normalize species names across tables/rasters."""  # docstring
    base = _canonicalize(name)  # base canonical form
    return SPECIES_NAME_ALIASES.get(base, base)  # apply alias map


def _species_raster_map(raster_stems: list[str]) -> dict[str, str]:  # map species to raster stem
    """Map canonical species names to raster stems for matching."""  # docstring
    return {_canonicalize(stem): stem for stem in raster_stems}  # build map


def extract_raster_stats(result_row: pd.Series, tifs: Sequence[Path]) -> dict[str, dict[str, float]]:
    """
    Extract per-raster median/variance from the zonal-stats row.
    This is tolerant of safe_colname() truncation and de-dup suffixes like _MD1/_VAR2.
    """
    cols = list(result_row.index)
    upper_cols = [str(c).upper() for c in cols]

    def _find_col(prefix: str, suffix: str) -> str | None:
        # exact
        exact = f"{prefix}{suffix}"
        if exact in upper_cols:
            return cols[upper_cols.index(exact)]

        # tolerant: startswith(prefix) and endswith(suffix or suffix+digits)
        suf = suffix
        for c, cu in zip(cols, upper_cols):
            if not cu.startswith(prefix):
                continue
            if cu.endswith(suf):
                return c
            if re.match(re.escape(prefix) + r".*" + re.escape(suf) + r"\d+$", cu):
                return c
        return None

    raster_stats: dict[str, dict[str, float]] = {}

    for tif_path in tifs:
        stem = tif_path.stem
        base = re.sub(r"[^A-Za-z0-9]", "", stem).upper() or "RAST"

        md_col = _find_col(base, "_MD")
        var_col = _find_col(base, "_VAR")

        if md_col is None and var_col is None:
            continue

        raster_stats[stem] = {
            "median": result_row.get(md_col) if md_col else None,
            "variance": result_row.get(var_col) if var_col else None,
        }

    return raster_stats


# Risk classes for TOTAL = A + B + C + D, where A and D are now true 0-1 scores.
# PROVISIONAL: set 2026-09-16 from 45 test squares on national forest/grassland across MT/ID/WY
# (all 14 species x 8 treatments). With these breaks about 37% of combinations are Very Low,
# 26% Low, 21% Medium, 12% High and 4% Very High. Confirm against known projects and
# pre/post-treatment field data before relying on them.
RISK_CLASS_BREAKS = (0.8, 1.0, 1.2, 1.4)
RISK_CLASS_LABELS = ("Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk")
NO_DATA_LABEL = "No data"


def classify_total_score(total_score: Optional[float]) -> Optional[str]:
    """Map a total A+B+C+D score to a risk class label."""
    if total_score is None or not math.isfinite(total_score):
        return None
    for brk, label in zip(RISK_CLASS_BREAKS, RISK_CLASS_LABELS):
        if total_score < brk:
            return label
    return RISK_CLASS_LABELS[-1]


def _finite_or_none(value) -> Optional[float]:
    """Return value as a finite float, or None for missing/NaN/inf."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def build_species_scores_from_rasters(  # combine A/B/C/D into scores
    raster_stats: dict[str, dict[str, float]],  # per-raster stats (0-1 scores from zonal_stats_batch)
    treatment: str,  # selected treatment
    c_col: str,  # Final Output column used for C
    species_table: pd.DataFrame,  # Updated Management Actions table
    final_output_df: pd.DataFrame,  # Final Output table
) -> pd.DataFrame:  # returns per-species results
    """
    Create per-species rows by combining raster scores with lookup values.

    A = habitat suitability score (0-1)            from the species raster
    B = treatment effect (0-1)                     from the Updated Management Actions table
    C = rate of proliferation                      from the Final Output table (unchanged)
    D = site vulnerability (0-1)                   mean of the HII, biodiversity-loss and
                                                   climate-velocity scores that have data
    """
    row = _action_row_for_treatment(species_table, treatment)  # find treatment row
    if row is None:  # no treatment row
        return pd.DataFrame()  # empty result
    species_cols = [c for c in species_table.columns if c not in SPECIES_META_COLS]  # species columns
    raster_map = _species_raster_map(list(raster_stats.keys()))  # map species to rasters
    species_keys = {_canonical_species_name(s) for s in species_cols}  # canonical species set
    final_map: dict[str, pd.Series] = {}  # Final Output rows by species
    if "Species" in final_output_df.columns:  # guard
        for _, row_data in final_output_df.iterrows():  # iterate rows
            key = _canonical_species_name(row_data.get("Species"))  # canonical key
            if key in species_keys:  # only keep relevant species
                final_map[key] = row_data  # store row

    # ---- D: site vulnerability = mean of the available 0-1 layer scores ----
    d_parts = {
        "HII": _finite_or_none(raster_stats.get(RESISTANCE_RASTER_KEYS["hii"], {}).get("median")),
        "Biodiversity": _finite_or_none(raster_stats.get(RESISTANCE_RASTER_KEYS["biodiversity"], {}).get("median")),
        "Clim_Velocity": _finite_or_none(raster_stats.get(RESISTANCE_RASTER_KEYS["clim_velocity"], {}).get("median")),
    }
    d_available = [v for v in d_parts.values() if v is not None]
    d_missing = [k for k, v in d_parts.items() if v is None]
    d_val = sum(d_available) / len(d_available) if d_available else None
    if not d_missing:
        d_note = ""
    elif d_available:
        d_note = f"No data for {', '.join(d_missing)}; D averaged from the other layer(s)."
    else:
        d_note = "No data for HII, Biodiversity or Clim_Velocity."

    rows = []  # output rows
    for species in species_cols:  # iterate species
        add_val = row.get(species)  # B from management actions table
        if pd.isna(add_val):  # skip missing values
            continue  # skip
        b_val = float(add_val)  # B = action effect
        canon_species = _canonical_species_name(species)  # canonical name
        alias = SPECIES_RASTER_ALIASES.get(canon_species)  # raster alias if any
        raster_key = raster_map.get(alias or canon_species)  # find raster stem
        stats = raster_stats.get(raster_key, {}) if raster_key else {}
        a_val = _finite_or_none(stats.get("median"))  # A = habitat suitability score (0-1)
        variance_val = _finite_or_none(stats.get("variance"))

        c_row = final_map.get(canon_species)  # row in Final Output
        c_val = None  # C value
        if c_row is not None and c_col in c_row:  # if C column exists
            c_val = _finite_or_none(c_row.get(c_col))

        missing_parts = []  # track missing components
        if not raster_key:
            missing_parts.append("A (no raster found for this species)")
        elif a_val is None:
            missing_parts.append("A (no habitat data in polygon: water or outside the model area)")
        if c_val is None:  # missing C
            missing_parts.append("C (Final Output)")
        if d_val is None:  # missing D
            missing_parts.append("D (HII/Biodiversity/Clim_Velocity)")

        total_score = None
        if not missing_parts:
            total_score = a_val + b_val + c_val + d_val  # A+B+C+D
        risk_class = classify_total_score(total_score) if total_score is not None else NO_DATA_LABEL

        rows.append(
            {
                "treatment": treatment,  # treatment name
                "species": species,  # species name
                "A_habitat": a_val,  # A value (0-1)
                "B_action": b_val,  # B value
                "C_proliferation": c_val,  # C value
                "D_resistance": d_val,  # D value (0-1)
                "D_note": d_note,  # why D is partial or missing
                "total_score": total_score,  # total score
                "risk_class": risk_class,  # risk class
                "median": a_val,  # kept for compatibility (same as A_habitat)
                "variance": variance_val,  # variance of the habitat score inside the polygon (0-1 scale)
                "missing_inputs": ", ".join(missing_parts),  # missing fields
            }
        )
    return pd.DataFrame(rows)  # return DataFrame


# -------------------- Per-pixel threat maps (A+B+C+D per cell) --------------------

class _GridTransform(tuple):
    """Minimal affine stand-in exposing .a/.b/.c/.d/.e/.f (GDAL-style: a=res_x, c=minx, f=maxy).

    The browser build has no rasterio.Affine; app.py only reads .a/.c/.f off this.
    """
    __slots__ = ()

    def __new__(cls, a, b, c, d, e, f):
        return super().__new__(cls, (a, b, c, d, e, f))

    a = property(lambda s: s[0])
    b = property(lambda s: s[1])
    c = property(lambda s: s[2])
    d = property(lambda s: s[3])
    e = property(lambda s: s[4])
    f = property(lambda s: s[5])


def compute_per_pixel_total(
    zone_geom,
    zones_crs,
    treatment: str,
    c_col: str,
    species_table: pd.DataFrame,
    final_output_df: pd.DataFrame,
    raster_dir=None,                 # kept for signature compatibility; ignored (R2 build)
    raster_patterns: Optional[Sequence[str]] = None,  # ignored
    species_names: Optional[Sequence[str]] = None,
    target_res: Optional[float] = None,               # ignored; grid fixed by the tiles
):
    """Per-pixel total threat (A/100 + B + C + D) for one polygon, per species.

    Returns (results, meta):
      results: dict[species_display_name] -> float32 array (NaN outside polygon/no-data)
      meta: {"transform", "crs", "shape", "bounds", "inside"} for ASCII-grid export.
    All layers share the pre-aligned R2 grid, so one window + one inside-mask serve every
    layer. D per pixel = mean of the available resistance scores (HII/50, 1-Bio/100,
    ClimVel/7). A/fill (0,127) and layer fill values are masked.
    """
    catalog = get_r2_catalog()
    grid_crs = catalog["crs"]
    stems = catalog_stems()
    smap = {re.sub(r"[^a-z0-9]", "", s.lower()): s for s in stems}

    geom_r = gpd.GeoSeries([zone_geom], crs=zones_crs).to_crs(grid_crs).iloc[0]
    minx, miny, maxx, maxy = geom_r.bounds

    win_tr_ref = None
    inside = None

    def _read(canon_key):
        """Fetch a stem's window; set the shared window transform + inside-mask on first read."""
        nonlocal win_tr_ref, inside
        stem = smap.get(canon_key)
        if stem is None:
            return None
        arr, win_tr = read_window_from_r2(stem, minx, miny, maxx, maxy, catalog)
        if win_tr_ref is None:
            win_tr_ref = win_tr
            inside = polygon_mask_on_window(geom_r, arr.shape, win_tr)
        return arr

    def _score_layer(key, fn, fill_zero):
        arr = _read(key)
        if arr is None:
            return None
        s = fn(arr)
        s = np.where(np.isnan(arr), np.nan, s)
        if fill_zero:
            s = np.where(arr == 0, np.nan, s)
        return s

    hii_s = _score_layer("hii", lambda a: np.clip(a / 50.0, 0.0, 1.0), fill_zero=False)
    bio_s = _score_layer("biodiversity", lambda a: np.clip(1.0 - a / 100.0, 0.0, 1.0), fill_zero=True)
    clim_s = _score_layer("climvelocity", lambda a: np.clip(a / 7.0, 0.0, 1.0), fill_zero=True)
    d_layers = [x for x in (hii_s, bio_s, clim_s) if x is not None]
    if d_layers:
        with np.errstate(invalid="ignore"):
            d_grid = np.nanmean(np.stack(d_layers, axis=0), axis=0)
    else:
        arr0 = _read(next(iter(smap)))  # no resistance layers: still establish the grid/mask
        d_grid = np.full(arr0.shape, np.nan, dtype="float32")

    row = _action_row_for_treatment(species_table, treatment)
    species_cols = [c for c in species_table.columns if c not in SPECIES_META_COLS]
    final_map = {}
    if "Species" in final_output_df.columns:
        for _, rd in final_output_df.iterrows():
            final_map[_canonical_species_name(rd.get("Species"))] = rd

    if species_names is None:
        species_names = species_cols

    results = {}
    for species in species_names:
        b_raw = row.get(species) if row is not None else None
        if b_raw is None or pd.isna(b_raw):
            continue
        b_val = float(b_raw)
        canon = _canonical_species_name(species)
        alias = SPECIES_RASTER_ALIASES.get(canon)
        sp_stem = smap.get(re.sub(r"[^a-z0-9]", "", str(alias or canon))) or smap.get(canon)
        if sp_stem is None:
            continue
        c_row = final_map.get(canon)
        c_val = None
        if c_row is not None and c_col in c_row and pd.notna(c_row.get(c_col)):
            c_val = float(c_row.get(c_col))
        if c_val is None:
            continue
        a_arr, _ = read_window_from_r2(sp_stem, minx, miny, maxx, maxy, catalog)
        a_s = np.clip(a_arr / 100.0, 0.0, 1.0)
        a_s = np.where(
            np.isnan(a_arr) | np.isin(a_arr, np.asarray(SPECIES_FILL_VALUES, dtype="float32")),
            np.nan, a_s,
        )
        total = a_s + b_val + c_val + d_grid
        total = np.where(inside, total, np.nan).astype("float32")
        results[species] = total

    tr = _GridTransform(win_tr_ref[0], 0.0, win_tr_ref[2], 0.0, win_tr_ref[4], win_tr_ref[5])
    meta = {
        "transform": tr,
        "crs": grid_crs,
        "shape": tuple(inside.shape),
        "bounds": (minx, miny, maxx, maxy),
        "inside": inside,
    }
    return results, meta


def array_to_asc_bytes(array, transform, crs=None) -> bytes:
    """Encode a 2D float array as an ESRI ASCII grid (.asc) — pure Python, GIS-importable.

    Replaces the old GeoTIFF export (rasterio is unavailable in the browser). The grid is
    georeferenced to the catalog CRS (EPSG:5070); NaN is written as the NODATA value -9999.
    """
    arr = np.asarray(array, dtype="float32")
    nrows, ncols = arr.shape
    cellsize = float(transform.a)
    xll = float(transform.c)
    yll = float(transform.f) - nrows * cellsize
    nodata = -9999.0
    filled = np.where(np.isfinite(arr), arr, nodata)
    header = (
        f"ncols {ncols}\n"
        f"nrows {nrows}\n"
        f"xllcorner {xll:.6f}\n"
        f"yllcorner {yll:.6f}\n"
        f"cellsize {cellsize:.10f}\n"
        f"NODATA_value {nodata:.0f}\n"
    )
    body = "\n".join(" ".join(f"{v:.6g}" for v in filled[r]) for r in range(nrows))
    return (header + body + "\n").encode("ascii")
