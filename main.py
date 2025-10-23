# app.py
import os, io, re, json, uuid, time, zipfile, tempfile, warnings
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, box
from shapely.ops import linemerge, unary_union
from pyproj import Transformer
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import httpx, folium, gdown

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="Commute Risk API (Drive + Sheets)", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        s.strip()
        for s in os.environ.get(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,https://*",
        ).split(",")
        if s.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ORS_API_KEY = os.environ.get(
    "ORS_API_KEY",
    # ta clé fournie (tu peux remplacer par une env avant déploiement)
    "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijg3ZWJiNWMzNGJlNDQwMjU4N2FkOWI3MGUzZWNmNDAzIiwiaCI6Im11cm11cjY0In0=",
).strip()
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"

# --- Liens Drive + Sheets fournis ---
DRIVE_FOLDER_URL = "https://drive.google.com/file/d/1fP2glevxgtzQdTgcLAsV2GsRzcYHj90W/view?usp=sharing"
SHEETS = {
    "vehicules": "https://docs.google.com/spreadsheets/d/1udeDqG8M_MR9bnDcneqB2M1ov7Ya2tqTjgurEweAvVM/export?format=csv",
    "caract": "https://docs.google.com/spreadsheets/d/1BhBVi0-D6bDZf03FBIlcvkTFVaCsw8RMtjqQKcFCn6U/export?format=csv",
    "lieux": "https://docs.google.com/spreadsheets/d/1pUPBGzYZ9RdGgA_VqLHWL7kI1k-1t2bXIzcN3TVIVFs/export?format=csv",
    "usagers": "https://docs.google.com/spreadsheets/d/1PM83BATrFyszWT6Ijy7TzFEj7l2RtWYxy2z17Cekd4o/export?format=csv",
}

CACHE = {
    "route500_shp": None,
    "accidents": None,
}
TMP_DIR = os.environ.get("DATA_DIR", "/tmp")
ROUTE500_DIR = os.path.join(TMP_DIR, "route500_cache")
os.makedirs(ROUTE500_DIR, exist_ok=True)


class RiskContext(BaseModel):
    lieux_multipliers: dict = Field(default_factory=dict)
    veh_multipliers: dict = Field(default_factory=dict)
    vma_thresholds: List[float] = Field(default_factory=lambda: [50, 70, 90, 110, 130])
    vma_multipliers: List[float] = Field(
        default_factory=lambda: [1.0, 1.05, 1.10, 1.15, 1.20]
    )


class CommuteBody(BaseModel):
    orig_lon: float
    orig_lat: float
    dest_lon: float
    dest_lat: float
    trips_per_year: int = 440
    k_paths_min_per_direction: int = 5
    time_cap_ratio: float = 1.30
    seg_len_m: float = 50.0
    buffer_routes_m: float = 120.0
    alt_geom_p: float = 0.9
    duration_target_min: float = 60.0
    eb_tau_km: float = 0.25
    context: Optional[RiskContext] = None


def _gdrive_folder_id(url: str) -> str:
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", url)
    if not m:
        raise ValueError("Drive folder URL invalide")
    return m.group(1)


def ensure_route500_shp() -> str:
    if CACHE["route500_shp"] and os.path.exists(CACHE["route500_shp"]):
        return CACHE["route500_shp"]
    folder_id = _gdrive_folder_id(DRIVE_FOLDER_URL)
    gdown.download_folder(
        id=folder_id, output=ROUTE500_DIR, quiet=True, use_cookies=False
    )
    shp_path = None
    for r, _, fs in os.walk(ROUTE500_DIR):
        for f in fs:
            lf = f.lower()
            if lf.endswith(".shp") and "tron" in lf:
                shp_path = os.path.join(r, f)
                break
        if shp_path:
            break
    if not shp_path:
        raise FileNotFoundError("Aucun shapefile TRONCON dans le dossier Drive")
    CACHE["route500_shp"] = shp_path
    return shp_path


def read_sheet_csv(url: str) -> pd.DataFrame:
    for sep in (",", ";", "\t"):
        try:
            return pd.read_csv(url, sep=sep, dtype=str, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(url, dtype=str, low_memory=False)


def parse_num_fr(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = (
        str(v)
        .replace("\u202f", "")
        .replace("\xa0", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    try:
        return float(s)
    except:
        return np.nan


def compute_lieux_multiplier(row, ctx: Optional[RiskContext]) -> float:
    if ctx is None:
        return 1.0
    m = 1.0
    if "catr" in row and pd.notna(row["catr"]):
        m *= float(ctx.lieux_multipliers.get(f"catr:{int(row['catr'])}", 1.0))
    if "surf" in row and pd.notna(row["surf"]):
        m *= float(ctx.lieux_multipliers.get(f"surf:{int(row['surf'])}", 1.0))
    if "circ" in row and pd.notna(row["circ"]):
        m *= float(ctx.lieux_multipliers.get(f"circ:{int(row['circ'])}", 1.0))
    if "vma" in row and pd.notna(row["vma"]):
        v = float(row["vma"])
        thr, mults = ctx.vma_thresholds, ctx.vma_multipliers
        idx = min([i for i, t in enumerate(thr) if v <= t] + [len(mults) - 1])
        m *= float(mults[idx])
    return float(m)


def compute_veh_multiplier(veh_counts: dict, ctx: Optional[RiskContext]) -> float:
    if ctx is None or not veh_counts:
        return 1.0
    tot = float(sum(veh_counts.values()))
    if tot <= 0:
        return 1.0
    s = 0.0
    for k, v in veh_counts.items():
        w = v / tot
        s += w * float(ctx.veh_multipliers.get(f"catv:{k}", 1.0))
    return float(max(0.3, min(3.0, s)))


def load_accidents(ctx: Optional[RiskContext]) -> gpd.GeoDataFrame:
    if CACHE["accidents"] is not None:
        return CACHE["accidents"]
    caract = read_sheet_csv(SHEETS["caract"])
    lieux = read_sheet_csv(SHEETS["lieux"])
    us = read_sheet_csv(SHEETS["usagers"])
    veh = read_sheet_csv(SHEETS["vehicules"])

    caract["lat"] = caract["lat"].map(parse_num_fr)
    caract["long"] = caract["long"].map(parse_num_fr)
    lieux["vma"] = lieux["vma"].map(parse_num_fr) if "vma" in lieux.columns else np.nan

    us["grav"] = pd.to_numeric(us["grav"], errors="coerce")
    sev_map = {1: 0, 2: 12, 3: 4, 4: 1}
    us["sev_w"] = us["grav"].map(sev_map).fillna(0).astype(float)
    us["injured_bool"] = us["grav"].isin([2, 3, 4])
    agg = us.groupby("Num_Acc", as_index=False).agg(
        sev_sum=("sev_w", "sum"), injured_any=("injured_bool", "any")
    )

    df = caract.merge(agg, on="Num_Acc", how="left").merge(
        lieux, on="Num_Acc", how="left"
    )
    df = df[(~df["lat"].isna()) & (~df["long"].isna())].copy()
    df["sev_sum"] = df["sev_sum"].fillna(0).infer_objects(copy=False)
    df["injured_any"] = df["injured_any"].fillna(False).astype(bool)

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["long"], df["lat"]), crs=4326
    )
    to2154 = Transformer.from_crs(4326, 2154, always_xy=True)
    X, Y = to2154.transform(gdf.geometry.x.values, gdf.geometry.y.values)
    gdf["geometry"] = gpd.points_from_xy(X, Y, crs=2154)

    if ctx is not None:
        gdf["lieux_mult"] = gdf.apply(
            lambda r: compute_lieux_multiplier(r, ctx), axis=1
        )
        veh_counts = (
            veh.groupby(["Num_Acc", "catv"]).size().reset_index(name="n")
            if {"Num_Acc", "catv"}.issubset(veh.columns)
            else pd.DataFrame()
        )
        vmap = {}
        if not veh_counts.empty:
            for num, sub in veh_counts.groupby("Num_Acc"):
                vmap[str(num)] = {
                    int(row["catv"]): int(row["n"])
                    for _, row in sub.iterrows()
                    if pd.notna(row["catv"])
                }
        gdf["veh_mult"] = gdf["Num_Acc"].map(
            lambda k: compute_veh_multiplier(vmap.get(str(k), {}), ctx)
        )
    else:
        gdf["lieux_mult"] = 1.0
        gdf["veh_mult"] = 1.0

    gdf["sev_sum_adj"] = (
        gdf["sev_sum"].astype(float)
        * gdf["lieux_mult"].astype(float)
        * gdf["veh_mult"].astype(float)
    )
    CACHE["accidents"] = gdf
    return gdf


def load_roads_bbox(
    shp_path: str, bbox: Tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    roads = gpd.read_file(shp_path, bbox=bbox)
    if roads.crs is None:
        roads = roads.set_crs(2154, allow_override=True)
    roads = (
        roads[roads.geometry.notna()].explode(index_parts=False).reset_index(drop=True)
    )
    roads["length_m"] = roads.length
    return roads


def split_line_fixed(line: LineString, L: float):
    res = []
    total = float(line.length)
    if total <= L:
        return [line]
    n = int(total // L)
    prev = 0.0
    for i in range(n):
        s = i * L
        e = (i + 1) * L
        p = line.interpolate(s)
        q = line.interpolate(e)
        res.append(LineString([p.coords[0], q.coords[0]]))
        prev = e
    if total > prev:
        p = line.interpolate(prev)
        q = line.interpolate(total)
        res.append(LineString([p.coords[0], q.coords[0]]))
    return res


def segments_from_lines(gdf: gpd.GeoDataFrame, seg_len: float):
    segs = []
    for geom in gdf.geometry:
        if isinstance(geom, LineString):
            segs += split_line_fixed(geom, seg_len)
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                segs += split_line_fixed(part, seg_len)
    if not segs:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs=gdf.crs)
    out = gpd.GeoDataFrame({"geometry": segs}, crs=gdf.crs)
    out["length_m"] = out.length
    return out


def sjoin_pts_to_segs(acc_pts: gpd.GeoDataFrame, seg_gdf: gpd.GeoDataFrame):
    if seg_gdf.empty:
        return seg_gdf.assign(sev_sum=0.0, acc_count=0, rk_raw=0.0, rk_0_100=0.0)
    pts = acc_pts[acc_pts["injured_any"] == True][
        ["Num_Acc", "sev_sum_adj", "geometry"]
    ].copy()
    seg = seg_gdf.copy()
    seg["sev_sum"] = 0.0
    seg["acc_count"] = 0
    if not pts.empty:
        try:
            j = gpd.sjoin_nearest(pts, seg, how="left", max_distance=18)
            agg = j.groupby(j.index_right, as_index=False).agg(
                sev_sum=("sev_sum_adj", "sum"), acc_count=("Num_Acc", "nunique")
            )
            seg.loc[agg["index_right"], "sev_sum"] = agg["sev_sum"].values
            seg.loc[agg["index_right"], "acc_count"] = agg["acc_count"].values
        except Exception:
            segb = seg.copy()
            segb["geometry"] = segb.buffer(12)
            j = gpd.sjoin(pts, segb, predicate="within", how="left")
            agg = j.groupby(j.index_right, as_index=False).agg(
                sev_sum=("sev_sum_adj", "sum"), acc_count=("Num_Acc", "nunique")
            )
            seg.loc[agg["index_right"], "sev_sum"] = agg["sev_sum"].values
            seg.loc[agg["index_right"], "acc_count"] = agg["acc_count"].values
    seg["rk_raw"] = seg["sev_sum"] / (seg["length_m"] / 1000.0 + 1e-6)
    return seg


def empirical_bayes_scale(seg_df: gpd.GeoDataFrame, tau_km: float) -> gpd.GeoDataFrame:
    df = seg_df.copy()
    if df.empty:
        return df.assign(rk_0_100=0.0, rk_eb=0.0)
    Lkm = (df["length_m"] / 1000.0).clip(1e-6, None)
    global_rate = df["sev_sum"].sum() / (Lkm.sum())
    alpha = (Lkm / (Lkm + tau_km)).values
    df["rk_eb"] = alpha * df["rk_raw"].values + (1 - alpha) * global_rate
    q10, q90 = np.percentile(df["rk_eb"].values, [10, 90])
    span = max(1e-6, q90 - q10)
    df["rk_0_100"] = ((df["rk_eb"] - q10) / span * 100.0).clip(0, 100)
    return df


def route_risk_index(route_geom, segs_df: gpd.GeoDataFrame) -> float:
    line = (
        route_geom
        if isinstance(route_geom, LineString)
        else linemerge(list(route_geom.geoms))
    )
    if line.length <= 0:
        return 0.0
    n = max(1, int(line.length // 50))
    pts = [line.interpolate(i * 50) for i in range(1, n + 1)]
    pts_g = gpd.GeoDataFrame(geometry=pts, crs=segs_df.crs)
    try:
        j = gpd.sjoin_nearest(
            pts_g, segs_df[["rk_0_100", "geometry"]], how="left", max_distance=40
        )
        vals = j["rk_0_100"].fillna(0).values
    except Exception:
        vals = np.zeros(len(pts), dtype=float)
    return float(np.mean(vals)) if len(pts) else 0.0


def exposure_stats(route_geom, segs_df: gpd.GeoDataFrame):
    line = (
        route_geom
        if isinstance(route_geom, LineString)
        else linemerge(list(route_geom.geoms))
    )
    n = max(1, int(line.length // 50))
    pts = [line.interpolate(i * 50) for i in range(1, n + 1)]
    if not pts:
        return 0.0, 0.0, 0.0
    gpts = gpd.GeoDataFrame(geometry=pts, crs=segs_df.crs)
    try:
        j = gpd.sjoin_nearest(
            gpts, segs_df[["rk_0_100", "geometry"]], how="left", max_distance=40
        )
        vals = j["rk_0_100"].fillna(0).values
    except Exception:
        vals = np.zeros(len(pts), dtype=float)
    below = (vals < 50).sum() * 50.0
    above = (vals >= 50).sum() * 50.0
    return below, above, float(vals.mean())


def ors_headers(k: str):
    return {"Authorization": k, "Content-Type": "application/json"}


async def ors_get_routes(o, d, k_paths: int, key: str):
    payload = {
        "coordinates": [list(o), list(d)],
        "alternative_routes": {"target_count": int(max(1, min(3, k_paths - 1)))},
    }
    async with httpx.AsyncClient(timeout=40) as client:
        r = await client.post(ORS_URL, headers=ors_headers(key), json=payload)
        r.raise_for_status()
        data = r.json()
    feats = data.get("features", [])
    out = []
    to2154 = Transformer.from_crs(4326, 2154, always_xy=True)
    for f in feats:
        g = f.get("geometry", {})
        if g.get("type") != "LineString":
            continue
        coords = g.get("coordinates", [])
        if not coords:
            continue
        xs, ys = [], []
        for lon, lat in coords:
            x, y = to2154.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        line = LineString(list(zip(xs, ys)))
        props = f.get("properties", {})
        summ = props.get("summary", {})
        dur = float(summ.get("duration", 0.0))
        dst = float(summ.get("distance", 0.0))
        out.append({"geom2154": line, "time_sec": dur, "distance_m": dst})
    return sorted(out, key=lambda r: r["time_sec"])


def geom_envelope_for_routes(geoms: List[LineString], pad_m: float = 120.0):
    u = unary_union(geoms)
    buf = u.buffer(pad_m)
    minx, miny, maxx, maxy = buf.bounds
    return (minx, miny, maxx, maxy), buf


def make_alt_probs_geometric(n: int, p: float) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    w = (1.0 - p) ** idx * p
    return w / np.sum(w)


def color_palette(n: int):
    base = [
        "#ff5252",
        "#40c4ff",
        "#ffd740",
        "#69f0ae",
        "#7c4dff",
        "#ff8a65",
        "#64b5f6",
        "#81c784",
        "#f06292",
        "#ba68c8",
    ]
    if n <= len(base):
        return base[:n]
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out


def make_map(o_wgs, d_wgs, seg_gdf, forward_routes, backward_routes):
    m = folium.Map(
        location=[o_wgs[1], o_wgs[0]], zoom_start=11, tiles="cartodbpositron"
    )
    if not seg_gdf.empty:
        segs_wgs = seg_gdf.to_crs(4326)

        def color_for(v):
            if v < 12:
                return "#2ecc71"
            if v < 25:
                return "#a3e4d7"
            if v < 40:
                return "#f4d03f"
            if v < 60:
                return "#e67e22"
            if v < 80:
                return "#d35400"
            return "#c0392b"

        folium.GeoJson(
            segs_wgs[
                ["rk_0_100", "acc_count", "sev_sum", "geometry"]
            ].__geo_interface__,
            style_function=lambda f: {
                "color": color_for(f["properties"]["rk_0_100"]),
                "weight": 6,
                "opacity": 0.9,
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=["rk_0_100", "acc_count", "sev_sum"],
                aliases=["Score", "Accidents", "Sévérité"],
                localize=True,
            ),
            name="Segments 50m",
        ).add_to(m)
    pal_f = color_palette(len(forward_routes))
    for idx, r in enumerate(forward_routes):
        g = gpd.GeoSeries([r["geom2154"]], crs=2154).to_crs(4326).iloc[0]
        folium.PolyLine(
            [(y, x) for x, y in g.coords],
            color=pal_f[idx],
            weight=3.0,
            opacity=0.95,
            tooltip=f"Aller alt {idx+1} • {r['time_sec']/60:.1f} min",
        ).add_to(m)
    pal_b = color_palette(len(backward_routes))
    for idx, r in enumerate(backward_routes):
        g = gpd.GeoSeries([r["geom2154"]], crs=2154).to_crs(4326).iloc[0]
        folium.PolyLine(
            [(y, x) for x, y in g.coords],
            color=pal_b[idx],
            weight=3.0,
            opacity=0.95,
            tooltip=f"Retour alt {idx+1} • {r['time_sec']/60:.1f} min",
            dash_array="8,6",
        ).add_to(m)
    folium.Marker(
        [o_wgs[1], o_wgs[0]], icon=folium.Icon(color="green"), tooltip="Origine"
    ).add_to(m)
    folium.Marker(
        [d_wgs[1], d_wgs[0]], icon=folium.Icon(color="red"), tooltip="Destination"
    ).add_to(m)
    return m._repr_html_()


@app.get("/health")
def health():
    ok_key = bool(ORS_API_KEY)
    ok_drive = True
    return {
        "status": "ok" if ok_key and ok_drive else "missing_data",
        "ors_key": ok_key,
        "drive_folder": DRIVE_FOLDER_URL,
    }


@app.post("/risk/commute")
async def risk_commute(req: Request, body: CommuteBody):
    try:
        key = (
            req.headers.get("Authorization")
            or req.headers.get("X-ORS-Key")
            or ORS_API_KEY
        ).strip()
        if not key:
            return JSONResponse(
                status_code=400, content={"error": "ORS_API_KEY manquant"}
            )
        acc = load_accidents(body.context)
        shp = ensure_route500_shp()

        o = (body.orig_lon, body.orig_lat)
        d = (body.dest_lon, body.dest_lat)
        fwd = await ors_get_routes(o, d, max(2, body.k_paths_min_per_direction), key)
        bwd = await ors_get_routes(d, o, max(2, body.k_paths_min_per_direction), key)
        if not fwd and not bwd:
            return JSONResponse(
                status_code=502, content={"error": "ORS n'a renvoyé aucun itinéraire"}
            )

        keep = lambda routes: (
            routes
            if not routes
            else [
                r
                for r in routes
                if r["time_sec"] <= body.time_cap_ratio * routes[0]["time_sec"]
            ]
            or [routes[0]]
        )
        fwd = keep(fwd)
        bwd = keep(bwd)

        all_lines = [r["geom2154"] for r in (fwd + bwd)]
        bbox, union_buf = geom_envelope_for_routes(
            all_lines, pad_m=body.buffer_routes_m
        )
        roads = load_roads_bbox(shp, bbox)
        if not roads.empty:
            roads = roads[roads.intersects(union_buf)].reset_index(drop=True)

        segs = (
            segments_from_lines(roads, body.seg_len_m)
            if not roads.empty
            else gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs=2154)
        )
        acc_corr = acc[acc.within(union_buf)].copy()
        if not segs.empty:
            segs = sjoin_pts_to_segs(acc_corr, segs)
            segs = empirical_bayes_scale(segs, body.eb_tau_km)

        def analyze(routes):
            if not routes:
                return {
                    "baseline_time_sec": None,
                    "expected_time_min": None,
                    "expected_route_risk_index_0_100": None,
                    "exposure": {
                        "below_50_km": 0,
                        "above_50_km": 0,
                        "mean_risk_along_best": 0,
                    },
                    "alternatives": [],
                }
            times = [r["time_sec"] for r in routes]
            probs = make_alt_probs_geometric(
                len(routes), max(1e-4, min(0.999, body.alt_geom_p))
            )
            rks = []
            alts = []
            for r, p in zip(routes, probs):
                rk = route_risk_index(r["geom2154"], segs) if not segs.empty else 0.0
                rks.append((p, rk))
                alts.append(
                    {
                        "prob": float(p),
                        "time_sec": float(r["time_sec"]),
                        "risk_index_0_100": float(rk),
                    }
                )
            best = routes[0]
            below, above, mean = (
                exposure_stats(best["geom2154"], segs)
                if not segs.empty
                else (0.0, 0.0, 0.0)
            )
            expected_risk = float(sum(p * r for p, r in rks))
            return {
                "baseline_time_sec": float(times[0]),
                "expected_time_min": float(
                    sum(p * (t / 60.0) for p, t in zip(probs, times))
                ),
                "expected_route_risk_index_0_100": expected_risk,
                "exposure": {
                    "below_50_km": below / 1000.0,
                    "above_50_km": above / 1000.0,
                    "mean_risk_along_best": mean,
                },
                "alternatives": alts,
            }

        fwd_stats = analyze(fwd)
        bwd_stats = analyze(bwd)
        combined_expected = np.nanmean(
            [
                x
                for x in [
                    fwd_stats["expected_route_risk_index_0_100"],
                    bwd_stats["expected_route_risk_index_0_100"],
                ]
                if x is not None
            ]
        )
        annualized = (
            float(combined_expected) * (body.trips_per_year / 440.0)
            if np.isfinite(combined_expected)
            else None
        )

        token = uuid.uuid4().hex
        tmp = tempfile.gettempdir()
        seg_path = os.path.join(tmp, f"segments_{token}.geojson")
        if not segs.empty:
            segs.to_crs(4326).to_file(seg_path, driver="GeoJSON")
        fwd_path = os.path.join(tmp, f"fwd_{token}.geojson")
        bwd_path = os.path.join(tmp, f"bwd_{token}.geojson")
        gpd.GeoDataFrame(geometry=[r["geom2154"] for r in fwd], crs=2154).to_crs(
            4326
        ).to_file(fwd_path, driver="GeoJSON")
        gpd.GeoDataFrame(geometry=[r["geom2154"] for r in bwd], crs=2154).to_crs(
            4326
        ).to_file(bwd_path, driver="GeoJSON")

        html = make_map(o, d, segs, fwd, bwd)

        STORE[token] = {
            "html": html,
            "segments": seg_path if not segs.empty else None,
            "fwd": fwd_path,
            "bwd": bwd_path,
            "metrics": {
                "calibration": {
                    "ref_daily_commute_min": body.duration_target_min,
                    "trips_per_year_ref": 440,
                },
                "params": {
                    "trips_per_year": body.trips_per_year,
                    "seg_len_m": body.seg_len_m,
                    "buffer_routes_m": body.buffer_routes_m,
                    "time_cap_ratio": body.time_cap_ratio,
                    "k_paths_min_per_direction": body.k_paths_min_per_direction,
                    "alt_geom_p": body.alt_geom_p,
                    "eb_tau_km": body.eb_tau_km,
                },
                "directions": {"aller": fwd_stats, "retour": bwd_stats},
                "combined": {
                    "expected_route_risk_index_0_100": (
                        float(combined_expected)
                        if combined_expected is not None
                        else None
                    ),
                    "annualized_risk_index_0_100": (
                        float(annualized) if annualized is not None else None
                    ),
                },
            },
        }
        return {
            "token": token,
            "segments_geojson_url": f"/segments/{token}",
            "forward_geojson_url": f"/forward/{token}",
            "backward_geojson_url": f"/backward/{token}",
            "map_url": f"/map/{token}",
            "metrics_url": f"/metrics/{token}",
        }
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=e.response.status_code,
            content={"error": f"ORS {e.response.status_code}: {e.response.text[:300]}"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"{type(e).__name__}: {e}"}
        )


STORE: Dict[str, Dict] = {}


@app.get("/segments/{token}")
def get_segments(token: str):
    if token not in STORE:
        return JSONResponse(status_code=404, content={"error": "inconnu"})
    p = STORE[token]["segments"]
    if not p or not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "introuvable"})
    with open(p, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))


@app.get("/forward/{token}")
def get_forward(token: str):
    if token not in STORE:
        return JSONResponse(status_code=404, content={"error": "inconnu"})
    p = STORE[token]["fwd"]
    if not p or not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "introuvable"})
    with open(p, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))


@app.get("/backward/{token}")
def get_backward(token: str):
    if token not in STORE:
        return JSONResponse(status_code=404, content={"error": "inconnu"})
    p = STORE[token]["bwd"]
    if not p or not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "introuvable"})
    with open(p, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))


@app.get("/map/{token}")
def get_map(token: str):
    if token not in STORE:
        return JSONResponse(status_code=404, content={"error": "inconnu"})
    return HTMLResponse(STORE[token]["html"])


@app.get("/metrics/{token}")
def get_metrics(token: str):
    if token not in STORE:
        return JSONResponse(status_code=404, content={"error": "inconnu"})
    return STORE[token]["metrics"]
