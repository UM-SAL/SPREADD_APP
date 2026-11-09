# This file is a Streamlit web app that lets a user draw or upload a shape,
# then run a raster zonal statistics model on that shape.
# Plain-language glossary:
# - GeoDataFrame: a table (like a spreadsheet) that also has shapes.
# - GeoJSON / Shapefile / GeoPackage (GPKG): common file types for shapes.
# - CRS: the coordinate system that says how numbers map to the Earth.
# - EPSG:4326: the common latitude/longitude coordinate system.
# - Raster: a grid of pixels (like an image) that stores values.
# - Zonal stats: summarize raster values inside a shape (like average in a box).

import streamlit as st
st.set_page_config(layout="wide")

# Import built-in tools for working with the operating system.
import os
import glob
# Import a tool to create temporary folders that auto-clean themselves.
import tempfile
# Import a tool to open zip files.
import zipfile
import shutil
import re
# Import Optional for type hints that can be a value or None.
from typing import Optional
# Import math helpers (we use this for NaN checks).
import math

# Import Folium for interactive maps.
import folium
# Import GeoPandas for reading and handling geographic files.
import geopandas as gpd
# Import Pandas for table concatenation.
import pandas as pd
# Browser build: raster metadata/bounds come from the R2 catalog (no rasterio in Pyodide).
# Import Streamlit for the web app UI.
import streamlit as st
# Import the Folium drawing plugin so users can draw on the map.
from folium.plugins import Draw
# Import tools to convert between GeoJSON and Shapely shapes.
from shapely.geometry import mapping, shape as shapely_shape, Point
# Import the Streamlit-Folium connector to show Folium maps in Streamlit.
from streamlit_folium import st_folium

# Optional faster layer reader for multi-layer uploads (GPKG / GDB)
try:
    import pyogrio
    HAS_PYOGRIO = True
except Exception:
    HAS_PYOGRIO = False

DEFAULT_NODATA = None  # if rasters do not declare nodata, do not force-mask any value

from rebekahs_model import (
    load_csv_latin1,
    get_treatment_options,
    get_final_output_column_options,
    extract_raster_stats,
    build_species_scores_from_rasters,
    zonal_stats_gdf,
)
from r2_data import get_r2_catalog, catalog_stems, R2_BASE_URL

# -------------------- Helper functions section --------------------

SPECIES_TABLE_PATH = "Species_List 1(Updated Management Actions).csv"
FINAL_OUTPUT_PATH = "Species_List 1(Final Output).csv"
CITATIONS_PATH = "Species_List 1(Citations).csv"

# -------------------- Species priority / ordering --------------------
# Ordering by Montana Dept of Agriculture noxious weed priority (1A = highest
# regulatory threat ... "unlisted" = lowest), split into two growth-form groups.
# Used to order results columns, citations, and result plots so the highest-threat
# species appear first. EDITABLE — adjust rank to change ordering everywhere.
# NOTE: cheatgrass is only regulatory Priority 3 but a high ecological threat; bump
# its rank here if you want it earlier.
#   name: (group, MT priority label, rank)   -- lower rank = higher threat / shown first
SPECIES_PRIORITY = {
    # Forbs / broadleaf
    "Rush Skeletonweed": ("Forbs", "1A", 1),
    "Blueweed": ("Forbs", "1B", 2),
    "Common Buckthorn": ("Forbs", "2A", 3),
    "Spotted Knapweed": ("Forbs", "2B", 4),
    "Leafy Spurge": ("Forbs", "2B", 5),
    "Sulfur Cinquefoil": ("Forbs", "2B", 6),
    "Dalmatian Toadflax": ("Forbs", "2B", 7),
    "Yellow Toadflax": ("Forbs", "2B", 8),
    "Oxeye Daisy": ("Forbs", "2B", 9),
    "Whitetop": ("Forbs", "2B", 10),
    "Common Tansy": ("Forbs", "2B", 11),
    # Invasive annual grasses
    "Ventenata": ("Invasive Annual Grasses", "2A", 12),
    "Cheatgrass": ("Invasive Annual Grasses", "3", 13),
    "Japanese Brome": ("Invasive Annual Grasses", "unlisted", 14),
}
SPECIES_GROUP_ORDER = ["Forbs", "Invasive Annual Grasses"]
_SPECIES_PRIORITY_CANON = {
    re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in SPECIES_PRIORITY.items()
}


def species_rank(name) -> int:
    """Threat rank for a species display name (lower = higher threat). Unknown = 999."""
    return _SPECIES_PRIORITY_CANON.get(re.sub(r"[^a-z0-9]", "", str(name).lower()), (None, None, 999))[2]


def species_group(name) -> str:
    """Growth-form group ('Forbs' / 'Invasive Annual Grasses') for a species name."""
    return _SPECIES_PRIORITY_CANON.get(re.sub(r"[^a-z0-9]", "", str(name).lower()), ("Other", None, 999))[0]


CONFIDENCE_HELP = (
    "Confidence = how uniform this species' habitat suitability is across the cells "
    "inside your polygon. Low variability → High confidence (one risk value represents "
    "the whole polygon well); high variability → Low confidence (the polygon spans very "
    "different habitat). Note: cutoffs are currently relative to this run's results."
)
RISK_HELP = (
    "Risk class from the total score A+B+C+D — Very Low <0.8, Low <1.0, Medium <1.2, "
    "High <1.4, Very High ≥1.4 (breakpoints provisional)."
)


def _condensed_col_config(df):
    """Hover-help tooltips on the Risk / Confidence column headers (#confidence-help)."""
    cfg = {}
    for c in df.columns:
        if str(c).endswith("(Confidence)"):
            cfg[c] = st.column_config.Column(help=CONFIDENCE_HELP)
        elif str(c).endswith("(Risk)"):
            cfg[c] = st.column_config.Column(help=RISK_HELP)
    return cfg


@st.cache_data(show_spinner=False)
def load_species_table() -> pd.DataFrame:
    """Load the species lookup table with a permissive encoding."""
    return load_csv_latin1(SPECIES_TABLE_PATH)


@st.cache_data(show_spinner=False)
def load_final_output_table() -> pd.DataFrame:
    """Load the Final Output table with a permissive encoding."""
    return load_csv_latin1(FINAL_OUTPUT_PATH)


@st.cache_data(show_spinner=False)
def load_citations_table() -> pd.DataFrame:
    """Load citations table."""
    return load_csv_latin1(CITATIONS_PATH)

def _ensure_gdf_crs(gdf: gpd.GeoDataFrame, default_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Ensure a GeoDataFrame has a CRS; if missing, set a default (no reprojection)."""
    if gdf.crs is None:
        st.warning(f"Uploaded data has no CRS; assuming {default_crs}.")
        return gdf.set_crs(default_crs, allow_override=True)
    return gdf


def _to_display_4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy reprojected to EPSG:4326 for Folium display."""
    gdf = _ensure_gdf_crs(gdf, "EPSG:4326")
    if str(gdf.crs).upper() in ("EPSG:4326", "WGS84"):
        return gdf
    try:
        return gdf.to_crs("EPSG:4326")
    except Exception as exc:
        st.error(f"Could not reproject upload to EPSG:4326 for display: {exc}")
        return gdf


    
# -------------------- CONNECTION TO THE DATASET (Cloudflare R2) --------------------

# Import pathlib for easy path handling.
import pathlib

# Browser build: rasters are pre-tiled onto one grid and served from a public Cloudflare
# R2 bucket (see r2_data.py + prepare_r2_tiles.py). No Hugging Face / rasterio needed.
HF_DATASET_ID = "UniversityOfMontanaSAL/Full_Rasters"  # provenance only (shown in status)
# Diagram image ships in the repo root (mounted by index.html for stlite).
DIAGRAM_FILENAME = "SPREADD_diagram_v2.png"

st.title("S.P.R.E.A.D.D.")
st.markdown("#### Spatial Probability of Risk from Environmental Action Driven Dispersal")
st.markdown(
    "Upload your own shapefile or draw a bounding box to determine the estimated risk of introducing an invasive plant species if a management action (treatment) is taken."
)
# Cache this function so we do not hit the network every time.
@st.cache_data(show_spinner=False)
def check_dataset_connection():
    """Lightweight check that the R2 raster bucket is reachable; no data download."""
    catalog = get_r2_catalog()
    return {"sha": "R2", "files": list(catalog.get("stems", []))}

# Cache the raster catalog once per session.
@st.cache_resource
def get_raster_paths():
    """Return raster inventory from the R2 catalog.

    Keeps the old shape ({"dir", "tifs", "ovr"}) so the rest of app.py is unchanged. The
    'tifs' are lightweight pseudo-paths (``Path("<stem>.tif")``) whose ``.stem`` matches
    the zonal-stats column prefixes; no files are actually downloaded.
    """
    stems = catalog_stems()
    return {
        "dir": pathlib.Path("."),
        "tifs": [pathlib.Path(f"{s}.tif") for s in stems],
        "ovr": [],
    }


def _missing_required_rasters(raster_info):
    """Return list of missing resistance rasters used for D."""
    required = {"HII", "Biodiversity", "Clim_Velocity"}
    stems = {p.stem for p in raster_info.get("tifs", [])}
    return sorted(list(required - stems))
# -------------------- Dataset connection status --------------------

# Message is rendered at the very bottom of the page (see end of file).
dataset_status_md = None

# NOTE: the status text is computed at the very BOTTOM of this file, after all
# helper functions (e.g. _auto_detect_nodata) are defined. Computing it here raised
# a NameError that an inner `except: pass` silently swallowed (bug #5), so the nodata
# line and the missing-raster warning never appeared.
# Cache the raster extent so we only scan raster headers once.
@st.cache_data(show_spinner=False)
def get_raster_extent_4326():
    """Combined raster bounds in EPSG:4326 (lat/lon), read from the R2 catalog.

    All layers share one pre-aligned grid, so the catalog stores a single bounds_4326.
    """
    catalog = get_r2_catalog()
    bounds = catalog.get("bounds_4326")
    if not bounds:
        return None
    return {
        "bounds_4326": list(bounds),  # [west, south, east, north]
        "raster_count": len(catalog.get("stems", [])),
        "missing_crs": [],
    }
    
    
# -------------------- INPUT GEOMETRY HANDLING FUNCTIONS --------------------
# These functions handle shapefile/geojson/geopackage uploads and reading.

# Find the first .shp file inside a folder (recursively).
def _find_shapefile_path(directory: str) -> Optional[str]:
    """Return the first .shp file found under directory (recursively)."""
    # Walk through all folders and files inside the given directory.
    for root, _, files in os.walk(directory):
        # Look at every file name in this folder.
        for fname in files:
            # If the file is a .shp file, return its full path.
            if fname.lower().endswith(".shp"):
                return os.path.join(root, fname)
    # If no .shp file is found, return None.
    return None

# Find the first usable spatial dataset in an extracted upload directory.
def _find_data_path(extract_dir: str) -> Optional[str]:
    """Return path to a .gdb directory, .gpkg, or .shp inside extract_dir."""
    gdbs = glob.glob(os.path.join(extract_dir, "*.gdb"))
    if gdbs:
        return gdbs[0]

    gpkg = glob.glob(os.path.join(extract_dir, "*.gpkg"))
    if gpkg:
        return gpkg[0]

    shp = glob.glob(os.path.join(extract_dir, "**/*.shp"), recursive=True)
    if shp:
        return shp[0]

    return None

def _list_layers(path: str) -> list[str]:
    """List layers for GPKG/GDB when pyogrio is available."""
    if HAS_PYOGRIO:
        try:
            return [name for name, _ in pyogrio.list_layers(path)]
        except Exception:
            return []
    return []

def _read_layer(path: str, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read a vector layer, preferring pyogrio when present."""
    if HAS_PYOGRIO:
        try:
            return pyogrio.read_dataframe(path, layer=layer)
        except Exception:
            pass
    return gpd.read_file(path, layer=layer)

# Load an uploaded file into a GeoPandas GeoDataFrame.
def load_geodataframe_from_upload(uploaded_file) -> gpd.GeoDataFrame:
    """Read an uploaded shapefile/GeoJSON/GeoPackage into a GeoDataFrame."""
    # Make a temporary folder so we can safely unpack or write the file.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Store the lower-case name so we can test the extension.
        name = uploaded_file.name.lower()
        # If the uploaded file is a zip, unpack it.
        if name.endswith(".zip"):
            # Open the zip file.
            with zipfile.ZipFile(uploaded_file, "r") as zf:
                # Extract all files into the temp folder.
                zf.extractall(tmpdir)
            # If a shapefile exists, read it; otherwise read the whole folder.
            read_target = _find_shapefile_path(tmpdir) or tmpdir
        else:
            # For a single file (GeoJSON/GPKG), write it to disk.
            read_target = os.path.join(tmpdir, uploaded_file.name)
            # Open a file in write-binary mode.
            with open(read_target, "wb") as f:
                # Copy the uploaded bytes into the temp file.
                f.write(uploaded_file.getbuffer())

        # Let GeoPandas read the file or folder and return a GeoDataFrame.
        return gpd.read_file(read_target)

# Convert a geometry into raw coordinate lists.
def geometry_coordinates(geom):
    """Convert a shapely geometry to raw coordinates (keeps polygons nested)."""
    # If the geometry is empty or missing, return None.
    if geom is None or geom.is_empty:
        return None
    # Convert the geometry to a GeoJSON-like dict and return coordinates.
    return mapping(geom).get("coordinates")

# -------------------- Load model tables + build treatment options ONCE --------------------
try:
    species_table = load_species_table()
except Exception as exc:
    st.error(f"Unable to load species table: {exc}")
    st.stop()

base_treatments = get_treatment_options(species_table)
VALID_MODEL_TREATMENTS = base_treatments[:]  # only the real treatments (no placeholders)

# -------------------- Streamlit page header --------------------

st.subheader("Treatment Area(s)")
input_mode = st.radio(
    "Choose geometry input method",
    options=["Draw feature", "Upload feature"],
    horizontal=True,
    key="input_mode",
)

# -------------------- Upload a file (only if Upload feature selected) --------------------
uploaded_gdf = None
uploaded_gdf_4326 = None
uploaded_file = None

selected_upload_idx = st.session_state.get("upload_select")
selected_upload_geom = None              # original CRS geometry (for analysis)
selected_upload_geom_4326 = None         # EPSG:4326 geometry (for display)
selected_upload_coords = None
selected_upload_label = None
selected_upload_subset_gdf = None
selected_drawn_coords = None
selected_drawn_label = None
selected_drawn_geom_geojson = None

# Track selected polygons (by __fid__) for uploads
if "upload_selected_ids" not in st.session_state:
    st.session_state.upload_selected_ids = set()

if input_mode == "Upload feature":
    st.subheader("Upload a shapefile or polygon file")
    st.caption(
        "**Accepted formats:**\n"
        "- `.geojson` / `.json`\n"
        "- Zipped shapefile (`.zip`)\n"
        "- `.gpkg` (GeoPackage)\n"
        "- Zipped File Geodatabase (`.gdb.zip`)\n"
        "\n"
        "**Notes:**\n"
        "- Only polygon layers are displayed.\n"
        "- Data is reprojected to WGS84 (EPSG:4326) for mapping.\n"
        "- For best reliability, use **GeoPackage (`.gpkg`)**."
    )

    uploaded_file = st.file_uploader("Upload geometry", type=["zip", "geojson", "json", "gpkg"])

    if uploaded_file and st.session_state.get("upload_last_filename") != uploaded_file.name:
        st.session_state.upload_selected_ids = set()
        st.session_state["selected_treatment_df"] = None
        st.session_state["selected_treatment_map"] = {}
        st.session_state["upload_last_filename"] = uploaded_file.name

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                name = uploaded_file.name.lower()
                data_path = None

                if name.endswith(".gpkg"):
                    data_path = os.path.join(tmpdir, uploaded_file.name)
                    with open(data_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                elif name.endswith(".zip"):
                    zpath = os.path.join(tmpdir, uploaded_file.name)
                    with open(zpath, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    with zipfile.ZipFile(zpath, "r") as zf:
                        zf.extractall(tmpdir)
                    data_path = _find_data_path(tmpdir)
                else:
                    data_path = os.path.join(tmpdir, uploaded_file.name)
                    with open(data_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                if not data_path:
                    st.error("Could not find a .gdb, .gpkg, or .shp inside the upload.")
                else:
                    layer_choice = None
                    layer_names = []
                    if data_path.lower().endswith((".gdb", ".gpkg")) and HAS_PYOGRIO:
                        layer_names = _list_layers(data_path)
                        if layer_names:
                            layer_choice = st.selectbox(
                                "Choose a layer",
                                layer_names,
                                key=f"upload_layer_{uploaded_file.name}",
                            )

                    uploaded_gdf = _read_layer(data_path, layer=layer_choice)

            if uploaded_gdf is not None:
                uploaded_gdf = uploaded_gdf[uploaded_gdf.geometry.notna()].copy()
                uploaded_gdf = uploaded_gdf[uploaded_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

            if uploaded_gdf is None or uploaded_gdf.empty:
                st.error("No polygon features found in this upload.")
            else:
                uploaded_gdf = _ensure_gdf_crs(uploaded_gdf, "EPSG:4326")
                uploaded_gdf = uploaded_gdf.reset_index(drop=True)
                uploaded_gdf["__fid__"] = uploaded_gdf.index.astype(int)

                # Apply any saved attribute edits for this file (edit-table feature),
                # keyed by file name + feature count so stale edits don't leak across files.
                _edits = st.session_state.get("upload_attr_edits")
                if (
                    _edits
                    and _edits.get("file") == uploaded_file.name
                    and _edits.get("n") == len(uploaded_gdf)
                ):
                    try:
                        _edf = pd.DataFrame(_edits["data"]).set_index("__fid__")
                        for _col in _edf.columns:
                            if _col in uploaded_gdf.columns and _col not in ("geometry", "__fid__"):
                                uploaded_gdf[_col] = uploaded_gdf["__fid__"].map(_edf[_col])
                    except Exception:
                        pass

                # DISPLAY COPY in EPSG:4326 for Folium only
                uploaded_gdf_4326 = _to_display_4326(uploaded_gdf.copy())
                uploaded_gdf_4326 = uploaded_gdf_4326.reset_index(drop=True)
                uploaded_gdf_4326["__fid__"] = uploaded_gdf["__fid__"]

                st.success(f"Loaded {len(uploaded_gdf)} polygon feature(s) from {uploaded_file.name}.")
                _attr_df = uploaded_gdf.drop(columns="geometry", errors="ignore").copy()
                # stlite/Arrow can render st.data_editor blank when a column has a dtype it
                # can't serialize (FileGDB OBJECTID/Shape_* numerics, WKB-bytes fields, etc.).
                # These attributes are shown as text, so coerce everything (except the internal
                # id) to strings for safe rendering in the browser.
                for _c in _attr_df.columns:
                    if _c != "__fid__":
                        _attr_df[_c] = _attr_df[_c].astype("string").fillna("")
                _edited_flag = (
                    "upload_attr_edits" in st.session_state
                    and st.session_state["upload_attr_edits"].get("file") == uploaded_file.name
                )
                edit_mode = st.toggle(
                    "✏️ Edit attributes",
                    key="upload_edit_mode",
                    help="Edit attribute values (e.g. Unit Number) before running. Geometry is not editable here.",
                )
                if edit_mode:
                    _show_cols = [c for c in _attr_df.columns if c != "__fid__"]
                    _edited = st.data_editor(
                        _attr_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400,
                        num_rows="fixed",
                        column_order=_show_cols,  # hide internal __fid__ (kept in the return)
                        key="upload_attr_editor",
                    )
                    c1, c2 = st.columns(2)
                    if c1.button("💾 Save edits", key="upload_attr_save"):
                        st.session_state["upload_attr_edits"] = {
                            "file": uploaded_file.name,
                            "n": len(uploaded_gdf),
                            "data": _edited.to_dict("records"),
                        }
                        st.success("Edits saved — they'll be used for selection and the model run.")
                        st.rerun()
                    if _edited_flag and c2.button("↩️ Revert to original", key="upload_attr_revert"):
                        st.session_state.pop("upload_attr_edits", None)
                        st.rerun()
                else:
                    if _edited_flag:
                        st.caption("Showing your edited attributes. Toggle **Edit attributes** to change them.")
                    st.dataframe(
                        _attr_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400,
                    )
        except Exception as exc:
            st.error(f"Unable to read file: {exc}")
else:
    # keep uploader hidden
    pass


# -------------------- Map drawing/upload display --------------------
st.subheader("Map")

if input_mode == "Upload feature":
    # New polygon selection map (replaces legacy single-select map)
    if uploaded_gdf_4326 is not None and not uploaded_gdf_4326.empty:
        non_geom_cols = [c for c in uploaded_gdf_4326.columns if c not in ["geometry", "__fid__"]]
        default_fields = non_geom_cols[:8]
        tooltip_fields = st.multiselect(
            "Tooltip fields (hover)",
            options=non_geom_cols,
            default=default_fields,
            key="upload_tooltip_fields",
        )

        # Cache GeoJSON to avoid repeated serialization on reruns
        geojson_sig = (
            tuple(uploaded_gdf_4326["__fid__"]),
            tuple(tooltip_fields),
        )
        if st.session_state.get("upload_geojson_sig") != geojson_sig:
            gdf_disp = uploaded_gdf_4326.copy()
            try:
                gdf_disp["geometry"] = gdf_disp.geometry.simplify(0.00005, preserve_topology=True)
            except Exception:
                pass
            st.session_state["upload_geojson"] = gdf_disp[["__fid__", "geometry"] + tooltip_fields].to_json()
            st.session_state["upload_geojson_sig"] = geojson_sig

        geojson_data = st.session_state.get("upload_geojson")

        # OpenStreetMap (free, no API key). CartoDB/Carto basemaps now require an
        # API key and render an "API KEY REQUIRED" watermark instead of tiles.
        m = folium.Map(tiles="OpenStreetMap")
        bounds = uploaded_gdf_4326.total_bounds  # minx, miny, maxx, maxy
        if bounds is not None and len(bounds) == 4:
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        def style_fn(feature):
            fid = feature["properties"]["__fid__"]
            if fid in st.session_state.upload_selected_ids:
                return {"weight": 3, "fillOpacity": 0.5}
            return {"weight": 1, "fillOpacity": 0.2}

        gj = folium.GeoJson(
            data=geojson_data,
            name="polygons",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
        )
        gj.add_to(m)

        out = st_folium(
            m,
            use_container_width=True,
            height=550,
            key="map_upload_select",
            returned_objects=["last_clicked"],
        )

        last_clicked = out.get("last_clicked") if out else None
        if last_clicked:
            lat = last_clicked["lat"]
            lng = last_clicked["lng"]
            p = Point(lng, lat)
            hit = None

            try:
                sidx = uploaded_gdf_4326.sindex
                cand_idx = list(sidx.intersection((p.x, p.y, p.x, p.y)))
                candidates = uploaded_gdf_4326.iloc[cand_idx]
            except Exception:
                candidates = uploaded_gdf_4326

            for _, row in candidates.iterrows():
                if row.geometry.contains(p):
                    hit = int(row["__fid__"])
                    break

            if hit is not None and st.session_state.get("upload_last_hit") != hit:
                if hit in st.session_state.upload_selected_ids:
                    st.session_state.upload_selected_ids.remove(hit)
                else:
                    st.session_state.upload_selected_ids.add(hit)
                st.session_state["upload_last_hit"] = hit
                st.rerun()

        # In All-polygons mode the click-to-select panel is irrelevant (treatments
        # come from a mapped column), so hide it (#21). upload_polygon_mode persists
        # in session_state from the run-scope radio below.
        _show_selection = st.session_state.get("upload_polygon_mode", "Selected polygons") != "All polygons"
        if _show_selection:
            st.subheader("Selection")
            st.write(f"Selected polygons: **{len(st.session_state.upload_selected_ids)}**")

            if st.button("Clear selection", key="clear_upload_selection"):
                st.session_state.upload_selected_ids = set()
                st.session_state["selected_treatment_df"] = None
                st.session_state["selected_treatment_map"] = {}
                st.rerun()
        else:
            st.caption(
                "All-polygons mode: treatments come from the column you map below — "
                "no per-polygon selection needed."
            )

        if _show_selection and st.session_state.upload_selected_ids:
            sel = uploaded_gdf_4326[
                uploaded_gdf_4326["__fid__"].isin(st.session_state.upload_selected_ids)
            ].drop(columns="geometry")

            # Prefill treatments from a persistent __fid__ -> treatment store.
            # Edits are applied via the on_change callback below (which writes to this
            # store) instead of feeding the editor's own output back in as its data —
            # that feedback caused the first selection to revert to blank (bug #1).
            treat_store = dict(st.session_state.get("selected_treatment_map", {}))
            sel["__fid__"] = sel["__fid__"].astype(str)
            # Coerce display-only attribute columns to strings so st.data_editor renders in
            # stlite (Arrow rejects some FileGDB dtypes / WKB bytes -> blank grid). The
            # editable model_treatment column is added separately below.
            for _c in list(sel.columns):
                sel[_c] = sel[_c].astype("string").fillna("")
            sel["model_treatment"] = sel["__fid__"].map(treat_store).fillna("")
            st.session_state["_sel_treat_fid_order"] = sel["__fid__"].tolist()

            # Convenience: apply one treatment to every selected polygon (#22),
            # instead of setting the dropdown row by row.
            same_for_all = st.checkbox(
                "Use the same treatment for all selected polygons",
                key="apply_same_treatment",
            )
            if same_for_all:
                bulk_treat = st.selectbox(
                    "Treatment for all selected polygons",
                    options=[""] + VALID_MODEL_TREATMENTS,
                    key="apply_same_treatment_value",
                )
                store = dict(st.session_state.get("selected_treatment_map", {}))
                for _f in sel["__fid__"].tolist():
                    store[_f] = bulk_treat
                st.session_state["selected_treatment_map"] = store
                treat_store = store
                sel["model_treatment"] = sel["__fid__"].map(store).fillna("")

            # Per-polygon treatment assignment via plain selectboxes. (st.data_editor can
            # render as a blank grid under stlite/Pyodide, so we avoid it here.) The
            # "same for all" shortcut above still drives every polygon when enabled.
            order = sel["__fid__"].tolist()
            # Label each row with its first few attribute values, if the upload has any.
            _label_cols = [c for c in sel.columns if c not in ("__fid__", "model_treatment")]
            store = dict(st.session_state.get("selected_treatment_map", {}))
            _opts = [""] + VALID_MODEL_TREATMENTS
            st.caption("Assign a model treatment to each selected polygon:")
            for _f in order:
                _rowmatch = sel.loc[sel["__fid__"] == _f]
                if _label_cols and not _rowmatch.empty:
                    _lbl = " | ".join(str(_rowmatch.iloc[0][c]) for c in _label_cols[:3])
                else:
                    _lbl = f"Polygon {_f}"
                _cur = store.get(_f, "")
                _idx = _opts.index(_cur) if _cur in _opts else 0
                _val = st.selectbox(
                    _lbl or f"Polygon {_f}",
                    options=_opts,
                    index=_idx,
                    key=f"sel_treat_{_f}",
                    disabled=same_for_all,  # when on, the single dropdown above governs
                )
                if not same_for_all:
                    store[_f] = _val
            st.session_state["selected_treatment_map"] = store
            treat_store = store

            # Publish the current selection's treatments (from the store) for the
            # ready-check and run block.
            treat_store = dict(st.session_state.get("selected_treatment_map", {}))
            st.session_state["selected_treatment_df"] = pd.DataFrame(
                {
                    "__fid__": sel["__fid__"].tolist(),
                    "model_treatment": [treat_store.get(f, "") for f in sel["__fid__"].tolist()],
                }
            )
        elif _show_selection:
            st.info("Click polygons on the map to toggle them into the selection.")

        # Single-polygon convenience for downstream logic
        if len(st.session_state.upload_selected_ids) == 1:
            fid = next(iter(st.session_state.upload_selected_ids))
            subset_orig = uploaded_gdf[uploaded_gdf["__fid__"] == fid]
            subset_disp = uploaded_gdf_4326[uploaded_gdf_4326["__fid__"] == fid]
            if not subset_orig.empty:
                selected_upload_geom = subset_orig.geometry.iloc[0]
                selected_upload_coords = geometry_coordinates(selected_upload_geom)
            if not subset_disp.empty:
                selected_upload_geom_4326 = subset_disp.geometry.iloc[0]
                selected_upload_label = f"Uploaded polygon {fid}"
        else:
            selected_upload_geom = None
            selected_upload_geom_4326 = None
            selected_upload_coords = None
            selected_upload_label = None

        if uploaded_gdf is not None:
            selected_upload_subset_gdf = uploaded_gdf[
                uploaded_gdf["__fid__"].isin(st.session_state.upload_selected_ids)
            ].copy()

    else:
        st.info("Upload a polygon dataset to select features on the map.")
else:
    # Map for drawing (Draw tools enabled)
    m = folium.Map(location=[46.87, -113.99], zoom_start=12)

    Draw(
        draw_options={
            "rectangle": True,
            "polygon": True,
            # Explicitly disable tools the model can't use, so they aren't
            # silently ignored after the user draws them.
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    output = st_folium(m, use_container_width=True, height=500, key="map_draw_only")

    if "drawn_shapes" not in st.session_state:
        st.session_state.drawn_shapes = []

    # Rebuild the list from the map's authoritative current drawings each run.
    # Leaflet features have no stable id, so the old "replace by id" logic never
    # matched and every interaction appended a duplicate; it also never removed
    # shapes deleted on the map. output["all_drawings"] is the current set.
    if output and output.get("all_drawings") is not None:
        st.session_state.drawn_shapes = [
            feat
            for feat in output["all_drawings"]
            if feat.get("geometry", {}).get("type", "") in ("Polygon", "MultiPolygon")
        ]

    st.subheader("Drawn Shapes:")
    selected_drawn_coords = None
    selected_drawn_label = None
    selected_drawn_geom_geojson = None

    if st.session_state.drawn_shapes:
        option_range = list(range(len(st.session_state.drawn_shapes)))
        option_labels = [
            f"Polygon {idx + 1}"
            for idx in range(len(st.session_state.drawn_shapes))
        ]
        st.caption(
            f"**Run prediction models all {len(st.session_state.drawn_shapes)} drawn "
            "polygon(s)** (shown as zone_id 1…N in the results). This selector only "
            "chooses which one to preview below."
        )
        selected_idx = st.selectbox(
            "Pick a drawn feature to preview",
            option_range,
            format_func=lambda i: option_labels[i],
            key="drawn_select",
        )
        selected_shape = st.session_state.drawn_shapes[selected_idx]
        with st.expander("Raw GeoJSON", expanded=False):
            st.json(selected_shape)

        try:
            geom = selected_shape["geometry"]
            coords = geom["coordinates"]
            selected_drawn_coords = coords
            selected_drawn_geom_geojson = geom
            geom_type = geom.get("type", "Feature")
            selected_drawn_label = f"Drawn {geom_type} {selected_idx + 1}"
            #st.write("Coordinates from selected drawing:")
            #st.write(coords)
        except (KeyError, TypeError) as e:
            st.error(f"Error extracting coordinates: {e}")
    else:
        st.info("No shapes drawn yet.")

# -------------------- Select Unique ID -------------------------------

st.divider()
st.subheader("Model Inputs")

# -------------------- Defaults (avoid NameError on reruns) --------------------
uid_field = None          # unique polygon id field from upload
user_treatment_col = None   # treatment column from upload (bulk mode)
bulk_mode = False
upload_polygon_mode = "Single polygon"
treatment = None
zones_gdf = None

# -------------------- If upload mode: pick polygon ID, run scope, and bulk mapping --------------------
# A) DRAW mode => always single polygon
if input_mode == "Draw feature":
    upload_polygon_mode = "Single polygon"
    bulk_mode = False
    uid_field = "zone_id"         # downstream we create/use zone_id
    user_treatment_col = None     # not used in draw mode

    st.subheader("Treatment (drawn polygon)")
    # Don't pre-select a treatment — start blank so the user must choose (#21).
    default_treatment = st.session_state.get("treatment_single", "")
    if default_treatment not in ([""] + VALID_MODEL_TREATMENTS):
        default_treatment = ""

    treatment = st.selectbox(
        "Pick a treatment option",
        options=[""] + VALID_MODEL_TREATMENTS,
        index=([""] + VALID_MODEL_TREATMENTS).index(default_treatment) if default_treatment in ([""] + VALID_MODEL_TREATMENTS) else 0,
        key="treatment_select_draw",
    )
    treatment = treatment.strip() if isinstance(treatment, str) else treatment
    if treatment == "":
        treatment = None
    st.session_state["treatment_single"] = treatment or ""

# B) UPLOAD mode => user chooses single vs all polygons
elif input_mode == "Upload feature" and uploaded_gdf is not None and not uploaded_gdf.empty:
    st.subheader("Run scope for uploaded polygons")
    upload_polygon_mode = st.radio(
        "Run predictions for…",
        options=["Selected polygons", "All polygons"],
        index=0,
        horizontal=True,
        key="upload_polygon_mode",
        help=(
            "Selected polygons: choose polygons on the map and assign treatments per feature.\n\n"
            "All polygons: map a treatment column from your shapefile to model treatments."
        ),
    )
    bulk_mode = (upload_polygon_mode == "All polygons")

    # Select a UNIQUE ID field (needed for joining results back)
    st.subheader("Polygon ID")
    candidate_id_cols = [
        c for c in uploaded_gdf.columns
        if c not in ["geometry", "__fid__"] and uploaded_gdf[c].notna().any()
    ]
    if not candidate_id_cols:
        st.error("No attribute columns found (besides geometry) to use as a unique ID.")
        uid_field = None
    else:
        # auto-suggest common ID fields first
        preferred = ["user_zone_id", "unique_id", "UniqueID", "OBJECTID", "ObjectID", "FID", "id", "ID", "Name", "name", "uuid", "UUID"]
        suggested = next((c for c in preferred if c in candidate_id_cols), None)

        # fallback: column that is mostly unique
        if suggested is None:
            for c in candidate_id_cols:
                nunique = uploaded_gdf[c].nunique(dropna=True)
                if nunique >= max(2, int(len(uploaded_gdf) * 0.9)):
                    suggested = c
                    break

        # final fallback
        if suggested is None:
            suggested = candidate_id_cols[0]

        uid_field = st.selectbox(
            "Choose a unique ID column (recommended)",
            options=candidate_id_cols,
            index=candidate_id_cols.index(suggested) if suggested in candidate_id_cols else 0,
            help="This ID will be carried into outputs so you can join results back to your original polygons.",
            key="uid_field_select",
        )

        # warn-only uniqueness check
        nunique = uploaded_gdf[uid_field].nunique(dropna=True)
        if nunique != len(uploaded_gdf):
            st.warning(
                f"`{uid_field}` is not unique ({nunique} unique values for {len(uploaded_gdf)} polygons). "
                "Join will still work, but duplicates will share the same ID."
            )

    # In bulk mode, also select the treatment attribute column
    if bulk_mode:
        st.subheader("Treatment column in your upload")
        # Auto-suggest a treatment-like column instead of defaulting to the first (#21).
        _treat_pref = ["treatment", "treatments", "trt", "rx", "action", "mgmt", "management", "activity"]
        _treat_suggest = next(
            (c for c in candidate_id_cols if re.sub(r"[^a-z0-9]", "", str(c).lower()) in _treat_pref),
            None,
        ) or next(
            (c for c in candidate_id_cols if "treat" in str(c).lower() or "action" in str(c).lower()),
            None,
        )
        user_treatment_col = st.selectbox(
        "Column in your upload that contains treatment labels/codes",
        options=candidate_id_cols,
        index=candidate_id_cols.index(_treat_suggest) if _treat_suggest in candidate_id_cols else 0,
        help="We will map each unique value to one of the model treatments.",
        key="user_treatment_col_select",
    )
        st.session_state["user_treatment_col"] = user_treatment_col

        # Build stable list of unique values (strings)
        user_values = (
            uploaded_gdf[user_treatment_col]
            .dropna()
            .astype(str)
            .map(lambda x: x.strip())
            .unique()
            .tolist()
        )
        user_values = sorted(user_values)

        # Initialize / update mapping table in session_state
        if "bulk_mapping_df" not in st.session_state:
            st.session_state["bulk_mapping_df"] = pd.DataFrame(
                {"user_value": user_values, "model_treatment": [""] * len(user_values)}
            )
        else:
            existing = st.session_state["bulk_mapping_df"].copy()
            if "user_value" not in existing.columns:
                # in case an older version used different columns
                existing = pd.DataFrame({"user_value": user_values, "model_treatment": [""] * len(user_values)})

            existing["user_value"] = existing["user_value"].astype(str).str.strip()
            new_df = pd.DataFrame({"user_value": user_values})
            merged = new_df.merge(existing, on="user_value", how="left")
            merged["model_treatment"] = merged["model_treatment"].fillna("").astype(str)
            st.session_state["bulk_mapping_df"] = merged

        st.caption("Map each unique uploaded treatment value to a model treatment. All rows must be mapped to run.")

        edited = st.data_editor(
            st.session_state["bulk_mapping_df"],
            use_container_width=True,
            height=420,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "user_value": st.column_config.TextColumn("Your treatment value", disabled=True),
                "model_treatment": st.column_config.SelectboxColumn(
                    "Model treatment (required)",
                    options=[""] + VALID_MODEL_TREATMENTS,
                    required=True,
                ),
            },
            key="bulk_mapping_editor",
        )
        # Do NOT write `edited` back into "bulk_mapping_df" (the editor's own data
        # source): feeding a keyed editor's output back as its input drops the first
        # selection until you pick it twice (bug #1). Keep the base table stable and
        # publish the current edits under a separate key for downstream consumers.
        st.session_state["bulk_mapping_current"] = edited

        unmapped = edited[edited["model_treatment"].fillna("").astype(str).str.strip() == ""]
        if not unmapped.empty:
            st.warning(f"{len(unmapped)} value(s) still unmapped — you must map all values to run bulk mode.")

    else:
        # ---- Selected polygons per-feature treatments (handled above) ----
        st.caption("Assign treatments in the selection table above for each selected polygon.")

# Final Output table (for choosing C)
try:
    final_output_df = load_final_output_table()
except Exception as exc:
    st.error(f"Unable to load Final Output table: {exc}")
    final_output_df = None

final_output_options = get_final_output_column_options(final_output_df) if final_output_df is not None else []
if not final_output_options:
    c_col = None
    st.warning("No usable Final Output columns found.")
else:
    option_labels = [label for label, _ in final_output_options]
    label_to_col = {label: col for label, col in final_output_options}
    default_label = next((l for l in option_labels if l.startswith("Column B")), option_labels[0])
    # Dropdown removed for now — auto-select the default column instead of asking the user.
    # To restore the interactive picker, uncomment the st.selectbox block below
    # and remove the `selected_label = default_label` line.
    # selected_label = st.selectbox(
    #     "Final Output metric (C)",
    #     options=option_labels,
    #     index=option_labels.index(default_label),
    #     help="Select which Final Output column to use as C in the decision tree.",
    #     key="c_col_select",
    # )
    selected_label = default_label
    c_col = label_to_col[selected_label]
    
# -------------------- Ready-to-run + Run button --------------------
all_touched = st.checkbox(
    "Include any-touched pixels (all_touched)",
    value=False,
    help="Turn on for small polygon areas <50m^2. If on, count any pixel touched by the polygon; if off, only pixels whose centers fall inside.",
    key="all_touched",
)
chosen_nodata = st.session_state.get("auto_nodata_value")

selected_geometry_source = "Uploaded" if input_mode == "Upload feature" else "Drawn"

if bulk_mode:
    mapping_df = st.session_state.get("bulk_mapping_current")
    mapping_ok = (
        mapping_df is not None
        and not mapping_df.empty
        and "model_treatment" in mapping_df.columns
        and (mapping_df["model_treatment"].fillna("").astype(str).str.strip() != "").all()
    )
    ready_to_run = (
        uploaded_gdf is not None
        and not uploaded_gdf.empty
        and (uid_field is not None)
        and (user_treatment_col is not None)
        and mapping_ok
        and (c_col is not None)
    )
else:
    if input_mode == "Upload feature":
        if upload_polygon_mode == "Selected polygons":
            uid_ready = uid_field is not None
            selection_ready = selected_upload_subset_gdf is not None and not selected_upload_subset_gdf.empty
            treatment_df = st.session_state.get("selected_treatment_df")
            treat_ready = False
            if treatment_df is not None and selection_ready:
                # merge to ensure all selected polygons have treatments
                tmp_uid = selected_upload_subset_gdf[[uid_field, "__fid__"]].copy()
                tmp_uid[uid_field] = tmp_uid[uid_field].astype(str)
                # Align __fid__ dtype on both sides (selected_treatment_df stores it
                # as string), otherwise the merge raises an int64-vs-object error.
                tmp_uid["__fid__"] = tmp_uid["__fid__"].astype(str)
                treatment_df = treatment_df.copy()
                treatment_df["__fid__"] = treatment_df["__fid__"].astype(str)
                merged = tmp_uid.merge(treatment_df, on="__fid__", how="left")
                treat_ready = merged["model_treatment"].fillna("").astype(str).str.strip().ne("").all()

            ready_to_run = (
                selection_ready
                and uid_ready
                and treat_ready
                and (c_col is not None)
            )
        else:
            ready_to_run = (
                (selected_upload_geom is not None)
                and (treatment is not None)
                and (c_col is not None)
            )
    else:
        ready_to_run = (
            (selected_drawn_geom_geojson is not None)
            and (treatment is not None)
            and (c_col is not None)
        )

polygon_error_slot = st.empty()
existing_polygon_errors = st.session_state.get("polygon_modeling_errors", [])
if existing_polygon_errors:
    with polygon_error_slot.container():
        for msg in existing_polygon_errors:
            st.error(msg)

run_prediction = st.button("Run prediction", disabled=not ready_to_run, key="run_prediction")

# Show the risk decision tree diagram below the button. The image ships in the repo root.
def _find_decision_tree_image() -> str | None:
    candidates = [
        DIAGRAM_FILENAME,
        "Risk_Decision_Tree_3.png",
        "Risk Decision Tree 3.png",
        "Risk_Decision_Tree.png",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    # Fallback: any PNG in the repo that looks like the diagram / a risk decision tree.
    for png_path in pathlib.Path(".").glob("*.png"):
        name = png_path.name.lower().replace("_", " ")
        if ("spreadd" in name and "diagram" in name) or (
            "risk" in name and "decision" in name and "tree" in name
        ):
            return str(png_path)
    return None


def _auto_detect_nodata(tif_paths):
    """R2 tiles are already NaN-cleaned, so there is no separate nodata value to apply."""
    return DEFAULT_NODATA


def _get_bounds_by_stem(raster_info):
    """All R2 layers share one grid, so every stem has the same catalog bounds_4326."""
    try:
        bounds_4326 = tuple(get_r2_catalog()["bounds_4326"])  # (west, south, east, north)
    except Exception:
        return {}
    return {p.stem: bounds_4326 for p in raster_info.get("tifs", [])}


def _polygons_outside(raster_bounds, zones_gdf):
    if not raster_bounds:
        return []
    zs = zones_gdf.to_crs("EPSG:4326")
    outside = []
    for stem, (w, s, e, n) in raster_bounds.items():
        box = shapely_shape({"type": "Polygon", "coordinates": [[(w, s), (w, n), (e, n), (e, s), (w, s)]]})
        if not zs.geometry.intersects(box).any():
            outside.append(stem)
    return outside

def _expected_md_cols(result_cols, stem):
    """Return True if a median column for stem exists in zonal stats output."""
    return len(_matching_md_cols(result_cols, stem)) > 0


def _matching_md_cols(result_cols, stem):
    """Return list of median columns matching a raster stem."""
    ups = [str(c).upper() for c in result_cols]
    stem_up = re.sub(r"[^A-Za-z0-9]", "", stem).upper()
    matches = []
    for c, u in zip(result_cols, ups):
        if u.startswith(stem_up) and u.endswith("MD"):
            matches.append(c)
            continue
        if re.match(re.escape(stem_up) + r".*MD\d+$", u):
            matches.append(c)
    return matches


def _warn_missing_resistance_medians(result_gdf, all_touched=False):
    """Warn if required resistance median columns are absent after zonal stats."""
    required = ["HII", "Biodiversity", "Clim_Velocity"]
    found = [s for s in required if _expected_md_cols(result_gdf.columns, s)]
    missing = [s for s in required if s not in found]
    # (Removed the "Resistance medians found: …" caption — redundant with the
    # dataset-status line at the bottom of the page; #21.)
    if missing:
        st.warning(
            "We could not read values from these resistance rasters: "
            + ", ".join(missing)
            + ". This usually means those rasters are missing from the cache or your polygons are outside their mapped area."
        )
    # If columns exist but are all NaN, D will still be empty
    all_nan = []
    for stem in found:
        md_cols = _matching_md_cols(result_gdf.columns, stem)
        vals = result_gdf[md_cols].copy()
        if vals.isna().all().all():
            all_nan.append(stem)
    if all_nan:
        hint = (
            "Try using all touched polygons by turning on 'Include any-touched pixels (all_touched)', or draw a larger polygon."
            if not all_touched
            else "Even with all_touched on, this usually means the polygon is outside valid data for that raster."
        )
        if len(all_nan) >= len(found):
            st.warning(
                "No resistance layers captured valid cells for your polygon(s): "
                + ", ".join(all_nan)
                + ". Site vulnerability (D) can't be computed. "
                + hint
            )
        else:
            st.info(
                "These resistance layers had no cells in your polygon(s): "
                + ", ".join(all_nan)
                + ". Site vulnerability (D) is averaged from the resistance layers that do have data. "
                + hint
            )


def _warn_per_polygon_d_issues(result_gdf, uid_field, all_touched=False):
    """Warn with per-polygon reasons when D inputs are missing/invalid or all zero."""
    required = ["HII", "Biodiversity", "Clim_Velocity"]
    md_cols = {stem: (_matching_md_cols(result_gdf.columns, stem) or [None])[0] for stem in required}
    if any(md_cols[s] is None for s in required):
        return pd.DataFrame()

    def _finite_float(value):
        try:
            v = float(value)
        except Exception:
            return None
        return v if math.isfinite(v) else None

    rows = []
    for _, r in result_gdf.iterrows():
        vals = {}
        bad = []
        for stem in required:
            raw = r.get(md_cols[stem])
            vals[stem] = _finite_float(raw)
            if vals[stem] is None:
                bad.append(stem)

        # Site vulnerability (D) is now averaged from whichever resistance layers
        # have data, so a polygon only has an unusable D when ALL THREE are missing.
        # Partial coverage (e.g. only the coarse Clim_Velocity is missing) is fine.
        if len(bad) < len(required):
            continue
        reason = "No resistance data here: HII, Biodiversity and Clim_Velocity are all missing."

        rows.append(
            {
                uid_field: str(r.get(uid_field, "")),
                "HII_MD": vals["HII"],
                "Biodiversity_MD": vals["Biodiversity"],
                "Clim_Velocity_MD": vals["Clim_Velocity"],
                "D_issue": reason,
            }
        )

    if rows:
        issue_df = pd.DataFrame(rows)
        hint = (
            "Try turning on 'Include any-touched pixels (all_touched)' or using a slightly larger polygon."
            if not all_touched
            else "Even with all_touched on, this usually means those polygons only overlap zero/background pixels."
        )
        st.warning(
            "Some polygons have unusable or zero D_resistance inputs. "
            "See details below. " + hint
        )
        st.dataframe(issue_df, use_container_width=True, hide_index=True, height=260)
        return issue_df
    return pd.DataFrame()


def _build_polygon_modeling_error_messages(issue_df, uid_field, all_touched=False):
    """Build per-polygon error text for display above the Run button."""
    if issue_df is None or issue_df.empty:
        return []
    hint = (
        "Try a larger polygon, or set all_touched=True."
        if not all_touched
        else "Try a larger polygon and verify it overlaps raster data."
    )
    messages = []
    for _, r in issue_df.iterrows():
        pid = str(r.get(uid_field, "")).strip() or "(unknown id)"
        reason = str(r.get("D_issue", "")).strip()
        msg = (
            f"Polygon {pid}: Polygon too small or has no overlapping raster for modeling. "
            f"{hint}"
        )
        if reason:
            msg += f" Details: {reason}"
        messages.append(msg)
    return messages

def _results_input_signature():
    """Best-effort fingerprint of the inputs that affect a model run. Used to flag
    cached results as out of date when inputs change since the last run (bug #8)."""
    import json
    try:
        parts = {
            "input_mode": input_mode,
            "bulk_mode": bool(bulk_mode),
            "all_touched": bool(st.session_state.get("all_touched")),
            "c_col": c_col,
            "n_drawn": len(st.session_state.get("drawn_shapes") or []),
            "upload_ids": sorted(str(x) for x in st.session_state.get("upload_selected_ids", set())),
        }
        bm = st.session_state.get("bulk_mapping_current")
        if bm is not None:
            parts["bulk_map"] = bm.to_dict(orient="records")
        stt = st.session_state.get("selected_treatment_df")
        if stt is not None:
            parts["selected_treat"] = stt.to_dict(orient="records")
        if not bulk_mode:
            parts["treatment"] = st.session_state.get("treatment")
        return json.dumps(parts, sort_keys=True, default=str)
    except Exception:
        return None


if run_prediction:
    # Original user ID column name, restored in the output tables (bug #7).
    # Overridden below for uploaded polygons; stays "zone_id" for drawn shapes.
    id_display_name = "zone_id"
    # Show a small toast message at the top.
    st.toast("Prediction started.")

    if bulk_mode:
        # BULK: use selected polygons from the uploaded dataset
        if uploaded_gdf is None or uploaded_gdf.empty:
            st.error("Bulk mode requires an uploaded polygon dataset.")
            st.stop()
    
        if uid_field is None or uid_field not in uploaded_gdf.columns:
            st.error("Bulk mode requires selecting a unique ID column from your upload.")
            st.stop()
    
        if "geometry" not in uploaded_gdf.columns:
            st.error("Upload has no geometry column.")
            st.stop()

        # All polygons mode: use the full uploaded dataset
        source_gdf = uploaded_gdf.copy()
    
        # Keep only ID + geometry, drop empty geometries
        zones_gdf = source_gdf[[uid_field, "geometry"]].copy()
        zones_gdf = zones_gdf[~zones_gdf.geometry.is_empty & zones_gdf.geometry.notna()].copy()
    
        if zones_gdf.empty:
            st.error("All uploaded geometries are empty/null after filtering.")
            st.stop()
    
        # Normalize ID field and rename to zone_id for downstream consistency
        id_display_name = uid_field  # preserve the user's ID column name for outputs (#7)
        zones_gdf[uid_field] = zones_gdf[uid_field].astype(str)
        zones_gdf = zones_gdf.rename(columns={uid_field: "zone_id"})
        uid_field = "zone_id"
    
    else:
        # SINGLE: either uploaded selected polygon(s) OR drawn polygon
        if selected_geometry_source == "Uploaded":
            if upload_polygon_mode == "Selected polygons":
                if selected_upload_subset_gdf is None or selected_upload_subset_gdf.empty:
                    st.error("No uploaded polygons selected.")
                    st.stop()

                if uid_field is None:
                    st.error("Select a unique ID column before running.")
                    st.stop()

                uid_col = uid_field  # keep original name for mapping
                id_display_name = uid_col  # preserve the user's ID column name for outputs (#7)
                zones_gdf = selected_upload_subset_gdf[[uid_col, "geometry"]].copy()
                zones_gdf = zones_gdf[~zones_gdf.geometry.is_empty & zones_gdf.geometry.notna()].copy()
                zones_gdf[uid_col] = zones_gdf[uid_col].astype(str)
                zones_gdf = zones_gdf.rename(columns={uid_col: "zone_id"})
                uid_field = "zone_id"

                treatment_df = st.session_state.get("selected_treatment_df")
                if treatment_df is None:
                    st.error("Assign treatments to selected polygons before running.")
                    st.stop()

                uid_map = selected_upload_subset_gdf[[uid_col, "__fid__"]].rename(columns={uid_col: "zone_id"})
                # Align __fid__ dtype on both sides before merging (see ready-check).
                uid_map["__fid__"] = uid_map["__fid__"].astype(str)
                treatment_df = treatment_df.copy()
                treatment_df["__fid__"] = treatment_df["__fid__"].astype(str)
                treatment_with_uid = uid_map.merge(treatment_df, on="__fid__", how="left")
                if treatment_with_uid["model_treatment"].fillna("").astype(str).str.strip().eq("").any():
                    st.error("All selected polygons must have a model treatment.")
                    st.stop()

                selected_treat_by_uid = dict(
                    zip(
                        treatment_with_uid["zone_id"].astype(str),
                        treatment_with_uid["model_treatment"].astype(str).str.strip(),
                    )
                )

            else:
                if selected_upload_geom is None or selected_upload_geom.is_empty:
                    st.error("No uploaded polygon selected.")
                    st.stop()

                zones_gdf = gpd.GeoDataFrame(
                    {"zone_id": ["1"]},
                    geometry=[selected_upload_geom],
                    crs=uploaded_gdf.crs if uploaded_gdf is not None else None,
                )
                uid_field = "zone_id"
                selected_treat_by_uid = {"1": treatment}

        else:
            # Run ALL drawn polygons, not just the one selected in the viewer
            # dropdown. Each gets zone_id 1..N (matching the viewer numbering); the
            # single chosen treatment is applied to every drawn polygon.
            drawn_feats = st.session_state.get("drawn_shapes", []) or []
            geoms, ids = [], []
            for i, feat in enumerate(drawn_feats, start=1):
                try:
                    g = shapely_shape(feat.get("geometry"))
                except Exception:
                    g = None
                if g is None or g.is_empty:
                    continue
                geoms.append(g)
                ids.append(str(i))
            if not geoms:
                st.error("No drawn geometry available.")
                st.stop()
            zones_gdf = gpd.GeoDataFrame(
                {"zone_id": ids},
                geometry=geoms,
                crs="EPSG:4326",
            )
            uid_field = "zone_id"
            selected_treat_by_uid = {i: treatment for i in ids}
    
    # CRS guard (after zones_gdf is built)
    if zones_gdf is None:
        st.error("Internal error: zones_gdf is None (no geometry was built).")
        st.stop()
    
    if zones_gdf.crs is None:
        st.warning("Input geometry has no CRS; assuming EPSG:4326.")
        zones_gdf = zones_gdf.set_crs("EPSG:4326", allow_override=True)

    if not isinstance(zones_gdf, gpd.GeoDataFrame):
        st.error(f"Internal error: zones_gdf is not a GeoDataFrame (got {type(zones_gdf)}).")
        st.stop()
    
    if zones_gdf.empty:
        st.error("Internal error: zones_gdf is empty (0 features). Select/draw a polygon and try again.")
        st.stop()
    
    if "geometry" not in zones_gdf.columns:
        st.error("Internal error: zones_gdf has no geometry column.")
        st.stop()
        
    # Keep source CRS for uploads so zonal_stats can correctly reproject to raster CRS.
    # For drawn polygons, CRS is already EPSG:4326 when created above.
    
    raster_info = get_raster_paths()
    if not raster_info["tifs"]:
        st.error("No .tif rasters found in the dataset cache.")
        st.stop()
    raster_bounds_by_stem = _get_bounds_by_stem(raster_info)
    
    raster_dir = str(raster_info["dir"])
    raster_patterns = ["**/*.tif", "**/*.tiff", "**/*.img", "**/*.bil"]
    
    with st.expander("Run diagnostics", expanded=False):
        st.caption(
            "Rasters served as pre-aligned ~98 m tiles from Cloudflare R2 (values are "
            "already NaN-cleaned; zonal statistics run in pure numpy in your browser)."
        )

    # Zonal statistics directly on the GeoDataFrame — fetches only the R2 tiles overlapping
    # each polygon (no files, no rasterio). Returns the same columns the old path produced.
    with st.spinner("Fetching raster tiles and computing zonal statistics…"):
        result_gdf = zonal_stats_gdf(zones_gdf, all_touched=all_touched)

    _warn_missing_resistance_medians(result_gdf, all_touched=all_touched)
    d_issue_df = _warn_per_polygon_d_issues(result_gdf, uid_field=uid_field, all_touched=all_touched)
    polygon_messages = _build_polygon_modeling_error_messages(
        d_issue_df,
        uid_field=uid_field,
        all_touched=all_touched,
    )
    st.session_state["polygon_modeling_errors"] = polygon_messages
    if polygon_messages:
        with polygon_error_slot.container():
            for msg in polygon_messages:
                st.error(msg)
    else:
        polygon_error_slot.empty()

    # warn if polygons are outside resistance raster footprints
    outside = _polygons_outside({k: v for k, v in raster_bounds_by_stem.items() if k in ["HII", "Biodiversity", "Clim_Velocity"]}, zones_gdf)
    if outside:
        st.warning(
            "Polygons do not intersect these resistance rasters: "
            + ", ".join(outside)
            + ". D_resistance will be empty."
        )

    st.success("Prediction complete.")
    
    mapping_df = st.session_state.get("bulk_mapping_current") if bulk_mode else None
    mapping_lookup = {}
    if bulk_mode:
        # Expect mapping_df columns: user_value, model_treatment
        if mapping_df is None or mapping_df.empty:
            st.error("Bulk mapping table is missing. Please map user treatments to model treatments.")
            st.stop()
    
        mapping_lookup = dict(
            zip(
                mapping_df["user_value"].astype(str).str.strip(),
                mapping_df["model_treatment"].astype(str).str.strip(),
            )
        )
    
    # Build a quick lookup from polygon id -> user treatment value (from uploaded attributes)
    user_treat_by_uid = {}
    if bulk_mode:
        user_treat_by_uid = dict(
            zip(
                zones_gdf[uid_field].astype(str),
                source_gdf[user_treatment_col].astype(str),
            )
        )
    elif selected_geometry_source == "Uploaded" and upload_polygon_mode == "Selected polygons":
        user_treat_by_uid = selected_treat_by_uid

    species_tables = []
    unmapped = []
    treat_by_zone = {}  # zone_id -> model treatment, for the per-pixel maps

    # result_gdf has one row per polygon (bulk) or one row (single)
    for _, row in result_gdf.drop(columns="geometry", errors="ignore").iterrows():
        uid_val = str(row.get(uid_field, ""))

        raster_stats = extract_raster_stats(row, raster_info["tifs"])

        if bulk_mode:
            user_val = str(user_treat_by_uid.get(uid_val, "")).strip()
            model_treat = mapping_lookup.get(user_val, "").strip()

            if not model_treat:
                unmapped.append({"uid": uid_val, "user_value": user_val})
                continue
            treat_by_zone[uid_val] = model_treat

            df_one = build_species_scores_from_rasters(
                raster_stats, model_treat, c_col, species_table, final_output_df
            )
            if not df_one.empty:
                df_one.insert(0, uid_field, uid_val)
                df_one.insert(1, "user_treatment_value", user_val)
                df_one.insert(2, "model_treatment", model_treat)
                species_tables.append(df_one)

        else:
            # selected polygons or drawn polygon use per-feature treatment
            model_treat = None
            if selected_geometry_source == "Uploaded" and upload_polygon_mode == "Selected polygons":
                model_treat = selected_treat_by_uid.get(uid_val, "").strip()
            else:
                model_treat = (treatment or "").strip() if treatment else ""

            if not model_treat:
                unmapped.append({"uid": uid_val, "user_value": model_treat})
                continue
            treat_by_zone[uid_val or "1"] = model_treat

            df_one = build_species_scores_from_rasters(
                raster_stats, model_treat, c_col, species_table, final_output_df
            )
            if not df_one.empty:
                df_one.insert(0, uid_field, uid_val or "1")
                if selected_geometry_source == "Uploaded" and upload_polygon_mode == "Selected polygons":
                    df_one.insert(1, "model_treatment", model_treat)
                species_tables.append(df_one)
    
    if unmapped:
        st.warning(f"{len(unmapped)} polygon(s) skipped because their treatment value was not mapped.")
        st.dataframe(pd.DataFrame(unmapped), use_container_width=True, hide_index=True)
    
    if not species_tables:
        st.error("No species results were produced (all polygons may have been unmapped or missing inputs).")
        # still show zonal stats output as fallback
        st.dataframe(result_gdf.drop(columns="geometry", errors="ignore"), use_container_width=True, hide_index=True)
        st.stop()
    
    species_df = pd.concat(species_tables, ignore_index=True)
    if "risk_class" in species_df.columns:
        species_df["risk_class"] = species_df["risk_class"].replace({"No Risk": "Very Low Risk"})

    # Turn the within-polygon spread of habitat suitability into a qualitative
    # "confidence" label (more spread = less confidence). Cutoffs are still per-run
    # tertiles, pending a decision on fixed cutoffs (#6/#16).
    variance_numeric = pd.to_numeric(species_df.get("variance"), errors="coerce")
    finite_variance = variance_numeric.dropna()
    finite_variance = finite_variance[finite_variance.map(math.isfinite)]
    if finite_variance.empty:
        species_df["variance_class"] = "Unknown"
    else:
        low_cut = float(finite_variance.quantile(0.33))
        high_cut = float(finite_variance.quantile(0.67))

        def _confidence_bin(v):
            if pd.isna(v) or not math.isfinite(float(v)):
                return "Unknown"
            if v <= low_cut:
                return "High confidence"
            if v <= high_cut:
                return "Moderate confidence"
            return "Low confidence"

        species_df["variance_class"] = variance_numeric.map(_confidence_bin)
    
    # Inputs / zonal stats (per polygon)
    inputs_df = result_gdf.drop(columns="geometry", errors="ignore").copy()
    
    # Condensed output:
    # - one row per polygon
    # - species as columns
    # - paired values = risk class and variance class
    risk_wide = species_df.pivot_table(
        index=uid_field,
        columns="species",
        values="risk_class",
        aggfunc="first",
    )
    variance_wide = species_df.pivot_table(
        index=uid_field,
        columns="species",
        values="variance_class",
        aggfunc="first",
    )
    species_order = list(dict.fromkeys(list(risk_wide.columns) + list(variance_wide.columns)))
    # Order species columns by Montana threat rank (highest threat first) (#priority).
    species_order = sorted(species_order, key=species_rank)
    condensed_df = pd.DataFrame(index=risk_wide.index.union(variance_wide.index))
    for species_name in species_order:
        condensed_df[f"{species_name} (Risk)"] = (
            risk_wide[species_name] if species_name in risk_wide.columns else None
        )
        condensed_df[f"{species_name} (Confidence)"] = (
            variance_wide[species_name] if species_name in variance_wide.columns else None
        )
    condensed_df = condensed_df.reset_index()
    
    # Expanded output (per polygon × per species), with plain-language column names
    # (#15). Drop the duplicate `median` column (identical to A_habitat).
    expanded_df = species_df.copy().drop(columns=["median"], errors="ignore")
    # Order rows by polygon, then by species threat rank (highest first) (#priority).
    if "species" in expanded_df.columns:
        expanded_df["_rank"] = expanded_df["species"].map(species_rank)
        _sort_cols = ([uid_field] if uid_field in expanded_df.columns else []) + ["_rank"]
        expanded_df = (
            expanded_df.sort_values(_sort_cols, kind="stable")
            .drop(columns="_rank")
            .reset_index(drop=True)
        )
    expanded_df = expanded_df.rename(
        columns={
            "treatment": "Treatment",
            "species": "Species",
            "A_habitat": "Habitat suitability (A)",
            "B_action": "Treatment effect (B)",
            "C_proliferation": "Spread potential (C)",
            "D_resistance": "Site vulnerability (D)",
            "total_score": "Total risk score",
            "risk_class": "Risk class",
            "variance_class": "Confidence",
            "variance": "Habitat variance",
            "D_note": "D note",
            "missing_inputs": "Missing inputs",
        }
    )

    # Restore the user's original ID column name in the output tables so results
    # join back to their shapefile on the field they uploaded (bug #7).
    out_uid_field = id_display_name if id_display_name else "zone_id"
    if out_uid_field != "zone_id":
        for _out_df in (condensed_df, expanded_df, inputs_df):
            if _out_df is not None and "zone_id" in _out_df.columns:
                _out_df.rename(columns={"zone_id": out_uid_field}, inplace=True)

    # cache results to keep them visible across reruns
    st.session_state["results_cache"] = {
        "uid_field": out_uid_field,
        "condensed_df": condensed_df,
        "expanded_df": expanded_df,
        "inputs_df": inputs_df,
        "signature": _results_input_signature(),
    }

    # Context for the per-pixel threat maps (survives reruns from the map selectors).
    try:
        st.session_state["viz_ctx"] = {
            "zones_geojson": zones_gdf[[uid_field, "geometry"]].to_crs("EPSG:4326").to_json(),
            "treat_by_zone": {str(k): v for k, v in treat_by_zone.items()},
            "c_col": c_col,
            "raster_dir": raster_dir,
        }
    except Exception:
        st.session_state.pop("viz_ctx", None)

    # UI display
    st.subheader("Results (Condensed)")
    st.dataframe(condensed_df, use_container_width=True, hide_index=True, column_config=_condensed_col_config(condensed_df))
    
    with st.expander("Expanded Output (per polygon × per species)", expanded=False):
        st.dataframe(expanded_df, use_container_width=True, hide_index=True)
    
    with st.expander("Expanded Inputs / Zonal Stats", expanded=False):
        st.dataframe(inputs_df, use_container_width=True, hide_index=True)
    
    # Single detailed download (expanded output + inputs together)
    detailed_report = expanded_df.merge(
        inputs_df,
        on=out_uid_field,
        how="left",
        suffixes=("_scores", "_inputs"),
    )
    st.download_button(
        "Download detailed report (CSV)",
        data=detailed_report.to_csv(index=False),
        file_name="detailed_report.csv",
        mime="text/csv",
        key="dl_detailed",
        help="Includes per-polygon × per-species scores with zonal inputs merged.",
    )
# If not running but cached results exist, display them so they persist across reruns
elif "results_cache" in st.session_state:
    cache = st.session_state["results_cache"]
    uid_field_cached = cache.get("uid_field", "zone_id")
    condensed_df = cache.get("condensed_df")
    expanded_df = cache.get("expanded_df")
    inputs_df = cache.get("inputs_df")

    cached_sig = cache.get("signature")
    if cached_sig is not None and cached_sig != _results_input_signature():
        st.warning(
            "⚠️ Inputs have changed since these results were generated — "
            "they may be out of date. Click **Run** to update."
        )

    if condensed_df is not None:
        st.subheader("Results (Condensed)")
        st.dataframe(condensed_df, use_container_width=True, hide_index=True, column_config=_condensed_col_config(condensed_df))

    if expanded_df is not None:
        with st.expander("Expanded Output (per polygon × per species)", expanded=False):
            st.dataframe(expanded_df, use_container_width=True, hide_index=True)

    if inputs_df is not None:
        with st.expander("Expanded Inputs / Zonal Stats", expanded=False):
            st.dataframe(inputs_df, use_container_width=True, hide_index=True)

    if expanded_df is not None and inputs_df is not None:
        detailed_report = expanded_df.merge(
            inputs_df,
            on=uid_field_cached,
            how="left",
            suffixes=("_scores", "_inputs"),
        )
        st.download_button(
            "Download detailed report (CSV)",
            data=detailed_report.to_csv(index=False),
            file_name="detailed_report.csv",
            mime="text/csv",
            key="dl_detailed",
            help="Includes per-polygon × per-species scores with zonal inputs merged.",
        )


# -------------------- Per-pixel threat maps (#19) --------------------
@st.cache_data(show_spinner=False)
def _cached_per_pixel(zone_wkt, zones_crs_str, treatment, c_col_v, raster_dir_v, species_tuple):
    """Cached per-pixel compute keyed by geometry/treatment/species so re-selecting
    species doesn't recompute unnecessarily."""
    from rebekahs_model import compute_per_pixel_total
    from shapely import wkt as _wkt

    geom = _wkt.loads(zone_wkt)
    species_table = load_species_table()
    final_output_df = load_final_output_table()
    return compute_per_pixel_total(
        geom,
        zones_crs_str,
        treatment,
        c_col_v,
        species_table,
        final_output_df,
        raster_dir_v,
        species_names=list(species_tuple),
    )


def render_per_pixel_maps():
    ctx = st.session_state.get("viz_ctx")
    if not ctx:
        return
    st.divider()
    st.subheader("Per-pixel threat maps")
    st.caption(
        "Total risk (A+B+C+D) per ~98 m cell inside your polygon, on a fixed 0–2 color "
        "scale so species are comparable. Class breaks: 0.8 / 1.0 / 1.2 / 1.4."
    )
    try:
        zones = gpd.read_file(ctx["zones_geojson"])
    except Exception:
        zones = None
    if zones is None or zones.empty:
        st.caption("No polygon geometry available for maps.")
        return
    zid_col = "zone_id" if "zone_id" in zones.columns else zones.columns[0]
    zone_ids = [str(z) for z in zones[zid_col].tolist()]

    if len(zone_ids) > 1:
        sel_zone = st.selectbox("Polygon to view", zone_ids, key="viz_zone_select")
    else:
        sel_zone = zone_ids[0]
    treat = ctx.get("treat_by_zone", {}).get(sel_zone, "")
    if not treat:
        st.caption("No treatment recorded for this polygon; re-run to enable maps.")
        return

    all_species = sorted(SPECIES_PRIORITY.keys(), key=species_rank)
    chosen = st.multiselect(
        "Species to show", options=all_species, default=all_species, key="viz_species_select"
    )
    if not chosen:
        st.caption("Select at least one species.")
        return

    col_a, col_b = st.columns(2)
    show_basemap = col_a.checkbox("Satellite basemap", value=True, key="viz_basemap")
    auto_scale = col_b.checkbox(
        "Auto-scale each map (show within-polygon detail)", value=False, key="viz_autoscale"
    )

    geom = zones.loc[zones[zid_col].astype(str) == sel_zone, "geometry"].iloc[0]
    with st.spinner("Computing per-pixel threat maps…"):
        try:
            results, meta = _cached_per_pixel(
                geom.wkt,
                str(zones.crs or "EPSG:4326"),
                treat,
                ctx["c_col"],
                ctx["raster_dir"],
                tuple(chosen),
            )
        except Exception as exc:
            st.error(f"Could not compute per-pixel maps: {exc}")
            return

    ordered = [s for s in chosen if s in results]
    if not ordered:
        st.caption("No per-pixel results (species may lack rasters or valid cells in this polygon).")
        return

    import io
    import zipfile

    import numpy as _np
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from rebekahs_model import array_to_asc_bytes

    ctx_mod = None
    if show_basemap:
        try:
            import contextily as ctx_mod
        except Exception:
            ctx_mod = None

    # Geographic extent of the grid (raster CRS), for overlaying on a basemap.
    tr = meta["transform"]
    res_x = tr.a
    h, w = meta["shape"]
    minx, maxy = tr.c, tr.f
    maxx, miny = minx + w * res_x, maxy - h * res_x
    extent = [minx, maxx, miny, maxy]
    try:
        crs_str = meta["crs"].to_wkt()
    except Exception:
        crs_str = str(meta["crs"])
    use_basemap = show_basemap and ctx_mod is not None
    alpha = 0.6 if use_basemap else 1.0

    ncols = 4
    nrows = int(_np.ceil(len(ordered) / ncols))
    fixed_norm = colors.Normalize(vmin=0.0, vmax=2.0)
    cmap = plt.get_cmap("YlOrRd")
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = _np.atleast_1d(axes).ravel()
    for j in range(len(ordered), len(axes)):
        axes[j].axis("off")
    for i, sp in enumerate(ordered):
        ax = axes[i]
        data = results[sp]
        if auto_scale:
            finite = data[_np.isfinite(data)]
            vmn = float(finite.min()) if finite.size else 0.0
            vmx = float(finite.max()) if finite.size else 1.0
            nrm = colors.Normalize(vmin=vmn, vmax=(vmx if vmx > vmn else vmn + 1e-6))
        else:
            nrm = fixed_norm
        ax.imshow(
            _np.ma.masked_invalid(data),
            cmap=cmap,
            norm=nrm,
            extent=extent,
            origin="upper",
            alpha=alpha,
            zorder=2,
            interpolation="nearest",
        )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        if use_basemap:
            try:
                ctx_mod.add_basemap(
                    ax, crs=crs_str, source=ctx_mod.providers.Esri.WorldImagery, attribution=False
                )
            except Exception:
                pass
        ax.set_title(sp, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    if auto_scale:
        st.caption("Each map is auto-scaled to its own min/max to reveal within-polygon variation (not comparable across species).")
    else:
        fig.colorbar(
            cm.ScalarMappable(norm=fixed_norm, cmap=cmap),
            ax=list(axes),
            fraction=0.02,
            pad=0.01,
            label="Total risk (A+B+C+D)",
        )
    st.pyplot(fig)
    plt.close(fig)

    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for sp in ordered:
                safe = re.sub(r"[^A-Za-z0-9]+", "_", sp)
                zf.writestr(
                    f"threat_zone{sel_zone}_{safe}.asc",
                    array_to_asc_bytes(results[sp], meta["transform"], meta["crs"]),
                )
            # Projection sidecar so GIS tools pick up the CRS of the .asc grids.
            zf.writestr("threat_grids.prj", 'PROJCS["NAD83 / Conus Albers",EPSG:5070]')
        st.download_button(
            "Download threat grids (ESRI ASCII .asc, .zip)",
            data=buf.getvalue(),
            file_name=f"per_pixel_threat_zone_{sel_zone}.zip",
            mime="application/zip",
            key="viz_dl_zip",
        )
    except Exception as exc:
        st.caption(f"Grid export unavailable: {exc}")

    # Interactive satellite view: contextily can't fetch tiles in the browser, so overlay
    # the threat grid on a Folium map (Esri World Imagery loads directly in the browser).
    try:
        import base64
        from shapely.geometry import box as _box

        st.markdown("**Interactive satellite view**")
        sp_map = st.selectbox(
            "Species to overlay on satellite imagery", ordered, key="viz_folium_species"
        )
        data = _np.ma.masked_invalid(results[sp_map])
        rgba = cmap(fixed_norm(data))  # (h, w, 4) floats, fixed 0-2 scale (matches grid above)
        rgba[..., 3] = _np.where(_np.isfinite(results[sp_map]), 0.75, 0.0)  # transparent where no data
        png_buf = io.BytesIO()
        plt.imsave(png_buf, (rgba * 255).astype("uint8"), format="png")
        data_uri = "data:image/png;base64," + base64.b64encode(png_buf.getvalue()).decode()

        # Grid bounds (EPSG:5070) -> lat/lon for the overlay. Small polygons: skew negligible.
        w, s, e, n = (
            gpd.GeoSeries([_box(minx, miny, maxx, maxy)], crs=meta["crs"])
            .to_crs("EPSG:4326")
            .total_bounds
        )
        fmap = folium.Map(tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", name="Satellite",
        ).add_to(fmap)
        folium.raster_layers.ImageOverlay(
            image=data_uri, bounds=[[s, w], [n, e]], opacity=0.75, name=sp_map
        ).add_to(fmap)
        fmap.fit_bounds([[s, w], [n, e]])
        st_folium(fmap, use_container_width=True, height=500, key="viz_folium_map")
        st.caption("Overlay is reprojected from the ~98 m grid; alignment is approximate.")
    except Exception as exc:
        st.caption(f"Satellite overlay unavailable: {exc}")


try:
    render_per_pixel_maps()
except Exception as _viz_exc:
    st.caption(f"Per-pixel maps unavailable: {_viz_exc}")

# -------------------- How the model works (decision tree) --------------------
def render_model_diagram():
    """Draw the current A+B+C+D model as a diagram (auto-matches the code constants)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    try:
        from rebekahs_model import (
            HII_MAX_CLASS,
            CLIM_VELOCITY_CAP_KM_YR,
            RISK_CLASS_BREAKS,
            RISK_CLASS_LABELS,
        )
    except Exception:
        HII_MAX_CLASS, CLIM_VELOCITY_CAP_KM_YR = 50.0, 7.0
        RISK_CLASS_BREAKS = (0.8, 1.0, 1.2, 1.4)
        RISK_CLASS_LABELS = (
            "Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk",
        )

    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#333", linewidth=1.2))
        ax.text(x + w / 2, y + h - 3, title, ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + h - 9, body, ha="center", va="top", fontsize=8)

    box(1, 62, 22, 34, "A · Habitat suitability", "species map ÷ 100\n(0–1, per pixel)", "#e8f4ea")
    box(25.5, 62, 22, 34, "B · Treatment effect", "from management\nlookup table\n(0–1)", "#e8eef7")
    box(50, 62, 22, 34, "C · Spread potential", "rate of proliferation\n(Final Output table)", "#f7f0e8")
    box(
        74.5, 62, 24.5, 34, "D · Site vulnerability",
        f"mean of 3 scores (0–1):\n• Human influence ÷ {HII_MAX_CLASS:g}\n"
        f"• Biodiversity lost = 1 − BII/100\n• Climate velocity ÷ {CLIM_VELOCITY_CAP_KM_YR:g}",
        "#f7e8ee",
    )

    box(30, 34, 40, 15, "TOTAL RISK", "A + B + C + D", "#efeff2")
    for cx in (12, 36.5, 61, 86.75):
        ax.annotate("", xy=(50, 49), xytext=(cx, 62), arrowprops=dict(arrowstyle="->", color="#777", lw=1))
    ax.annotate("", xy=(50, 24), xytext=(50, 34), arrowprops=dict(arrowstyle="->", color="#777", lw=1.3))

    labels = [l.replace(" Risk", "") for l in RISK_CLASS_LABELS]
    breaks = list(RISK_CLASS_BREAKS)
    edges = [breaks[0] - 0.2] + breaks + [breaks[-1] + 0.2]
    seg_colors = ["#1a9850", "#a6d96a", "#fee08b", "#fc8d59", "#d73027"]
    bar_x0, bar_w = 8, 84
    lo, hi = edges[0], edges[-1]
    for i in range(5):
        x = bar_x0 + (edges[i] - lo) / (hi - lo) * bar_w
        w = (edges[i + 1] - edges[i]) / (hi - lo) * bar_w
        ax.add_patch(Rectangle((x, 10), w, 6, facecolor=seg_colors[i], edgecolor="white"))
        ax.text(x + w / 2, 13, labels[i], ha="center", va="center", fontsize=7.5)
    for b in breaks:
        x = bar_x0 + (b - lo) / (hi - lo) * bar_w
        ax.plot([x, x], [9, 17], color="#333", lw=0.8)
        ax.text(x, 7.5, f"{b:g}", ha="center", va="top", fontsize=7)
    ax.text(50, 2, "Total score → risk class", ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    return fig


def _show_auto_diagram():
    try:
        import matplotlib.pyplot as _plt_diag

        _fig = render_model_diagram()
        st.pyplot(_fig)
        _plt_diag.close(_fig)
    except Exception as _diag_exc:
        st.caption(f"Model diagram unavailable: {_diag_exc}")


st.divider()
st.subheader("How the model works")
_diagram_path = _find_decision_tree_image()  # bundled PNG in the repo root
if _diagram_path:
    st.image(_diagram_path, use_container_width=True)
    with st.expander("Code-generated version (auto-matches the model constants)", expanded=False):
        _show_auto_diagram()
else:
    _show_auto_diagram()
st.caption(
    "Total risk = A + B + C + D (each 0–1; B and C from lookup tables). Higher "
    "biodiversity intactness lowers risk. Class breaks are provisional."
)

# -------------------- Citations --------------------

st.divider()
st.subheader("Citations")
try:
    citations_df = load_citations_table()
    if citations_df is not None and "Species" in citations_df.columns and "Sources" in citations_df.columns:
        # Grouped by species (#23).
        cdf = citations_df.dropna(subset=["Sources"]).copy()
        cdf["Species"] = cdf["Species"].fillna("General references").astype(str).str.strip()
        cdf["Sources"] = cdf["Sources"].astype(str).str.strip()
        present = [
            s for s in cdf["Species"].unique() if s not in ("General references", "Data layers")
        ]

        def _emit_species_block(sp):
            label = _SPECIES_PRIORITY_CANON.get(
                re.sub(r"[^a-z0-9]", "", str(sp).lower()), (None, None, None)
            )[1]
            header = f"**{sp}**" + (f" — MT priority {label}" if label else "")
            st.markdown(header)
            for src in cdf.loc[cdf["Species"] == sp, "Sources"].tolist():
                st.markdown(f"- {src}")

        # Two growth-form sections, species ordered by Montana threat rank (#priority).
        for group_name in SPECIES_GROUP_ORDER:
            members = sorted(
                [s for s in present if species_group(s) == group_name], key=species_rank
            )
            if not members:
                continue
            st.markdown(f"### {group_name}")
            for sp in members:
                _emit_species_block(sp)
        # Any species not in the priority config (safety net).
        leftover = sorted(s for s in present if species_group(s) == "Other")
        if leftover:
            st.markdown("### Other")
            for sp in leftover:
                _emit_species_block(sp)
        if "Data layers" in cdf["Species"].values:
            st.markdown("### Data layers (model inputs)")
            for src in cdf.loc[cdf["Species"] == "Data layers", "Sources"].tolist():
                st.markdown(f"- {src}")
            st.caption(
                "Canonical sources for each layer — confirm the exact dataset/version used "
                "(the model uses a recreated Human Influence Index)."
            )
        if "General references" in cdf["Species"].values:
            st.markdown("### General references")
            for src in cdf.loc[cdf["Species"] == "General references", "Sources"].tolist():
                st.markdown(f"- {src}")
    else:
        # Fallback for an older single-column citations file.
        sources = citations_df.get("Sources") if citations_df is not None else None
        if sources is not None:
            for src in sources.dropna().astype(str).tolist():
                st.markdown(f"- {src}")
        else:
            st.caption("No citations found.")
except Exception as exc:
    st.caption(f"Unable to load citations: {exc}")

# -------------------- Dataset connection status (bottom of page) --------------------
try:
    ds_status = check_dataset_connection()
    status_msg = f"✅ **Rasters connected:** Cloudflare R2 (`{len(ds_status['files'])}` layers)"
    try:
        raster_info_for_check = get_raster_paths()
        missing_req = _missing_required_rasters(raster_info_for_check)
        if missing_req:
            status_msg += f"  \n⚠️ Missing resistance rasters: `{', '.join(missing_req)}`"
        else:
            status_msg += "  \n`Resistance rasters:` HII, Biodiversity, Clim_Velocity"
    except Exception as raster_exc:
        status_msg += f"  \n`Raster check unavailable:` `{raster_exc}`"
    dataset_status_md = status_msg
except Exception as exc:
    dataset_status_md = f"⚠️ **Raster bucket not reachable:** `{R2_BASE_URL}` — {exc}"

if dataset_status_md:
    st.divider()
    st.markdown(dataset_status_md)
