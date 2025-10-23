import os, uuid, json, tempfile, warnings, asyncio
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from pyproj import Transformer
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.responses import ORJSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import httpx, folium
from sklearn.neighbors import KDTree

warnings.filterwarnings("ignore", category=UserWarning)

ORS_API_KEY = os.environ.get(
    "ORS_API_KEY",
    "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijg3ZWJiNWMzNGJlNDQwMjU4N2FkOWI3MGUzZWNmNDAzIiwiaCI6Im11cm11cjY0In0=",
).strip()
ORS_URL = os.environ.get(
    "ORS_URL", "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
)
ALLOWED_ORIGINS = [
    s.strip()
    for s in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://localhost:8080",
    ).split(",")
    if s.strip()
]

CARACT_CSV = os.environ.get("CARACT_CSV", "caract-2023.csv")
LIEUX_CSV = os.environ.get("LIEUX_CSV", "lieux-2023.csv")
VEHICULES_CSV = os.environ.get("VEHICULES_CSV", "vehicules-2023.csv")
USAGERS_CSV = os.environ.get("USAGERS_CSV", "usagers-2023.csv")

FAST_MODE = bool(int(os.environ.get("FAST_MODE", "1")))
EB_TAU_KM = float(os.environ.get("EB_TAU_KM", "0.20" if FAST_MODE else "0.25"))
SMOOTH_RADIUS_M = float(os.environ.get("SMOOTH_RADIUS_M", "80" if FAST_MODE else "120"))
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", "0.55" if FAST_MODE else "0.6"))
INTER_BETA = float(os.environ.get("INTER_BETA", "0.20" if FAST_MODE else "0.25"))
JOIN_RADIUS_M = float(os.environ.get("JOIN_RADIUS_M", "22" if FAST_MODE else "28"))

REF_DAILY_COMMUTE_MIN = float(os.environ.get("REF_DAILY_COMMUTE_MIN", "60"))
TRIPS_PER_YEAR_REF = float(os.environ.get("TRIPS_PER_YEAR_REF", "440"))
RISK_REF_MEDIAN_ENV = os.environ.get("RISK_REF_MEDIAN", "").strip()
RISK_REF_MEDIAN = float(RISK_REF_MEDIAN_ENV) if RISK_REF_MEDIAN_ENV else None
MINUTES_ANNUELLES_REF = REF_DAILY_COMMUTE_MIN * (TRIPS_PER_YEAR_REF / 2.0)

STATE = {"acc_gdf": None, "signature": None}
STORE: Dict[str, Dict] = {}
RESULT_CACHE: Dict[str, str] = {}
HTTP: Optional[httpx.AsyncClient] = None


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
    k_paths: int = 5
    time_cap_ratio: float = 1.28
    seg_len_m: float = 100.0 if FAST_MODE else 50.0
    corridor_m: float = 250.0 if FAST_MODE else 350.0
    alt_geom_p: float = 0.9
    hour: Optional[float] = None
    context: Optional[RiskContext] = None


def _sig(body: CommuteBody):
    r = lambda x: round(float(x), 4)
    return json.dumps(
        {
            "o": (r(body.orig_lon), r(body.orig_lat)),
            "d": (r(body.dest_lon), r(body.dest_lat)),
            "p": [
                int(max(5, body.k_paths)),
                round(body.time_cap_ratio, 3),
                round(body.seg_len_m, 1),
                round(body.corridor_m, 1),
                round(body.alt_geom_p, 4),
                None if body.hour is None else round(float(body.hour), 1),
            ],
        },
        sort_keys=True,
    )


def _parse_fr_float(v):
    if pd.isna(v):
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


def read_csv_cols(path, usecols=None):
    last = None
    for enc in ("utf-8", "cp1252", "latin1"):
        for sep in (";", "; ", ",", "\t"):
            try:
                return pd.read_csv(
                    path,
                    sep=sep,
                    encoding=enc,
                    usecols=usecols,
                    dtype={"Num_Acc": "string"},
                    low_memory=False,
                )
            except Exception as e:
                last = e
    raise RuntimeError(f"lecture csv impossible: {path} ({last})")


def compute_lieux_multiplier(row, ctx: Optional[RiskContext]):
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
        thr = ctx.vma_thresholds
        mults = ctx.vma_multipliers
        idx = 0
        for i, t in enumerate(thr):
            if v <= t:
                idx = i
                break
            idx = min(i + 1, len(mults) - 1)
        m *= float(mults[idx])
    return float(m)


def compute_veh_multiplier(veh_counts: dict, ctx: Optional[RiskContext]):
    if ctx is None or not veh_counts:
        return 1.0
    total = float(sum(veh_counts.values()))
    if total <= 0:
        return 1.0
    s = 0.0
    for k, v in veh_counts.items():
        s += (v / total) * float(ctx.veh_multipliers.get(f"catv:{k}", 1.0))
    return float(max(0.1, min(5.0, s)))


def load_accidents(ctx: Optional[RiskContext]):
    sig = (CARACT_CSV, LIEUX_CSV, VEHICULES_CSV, USAGERS_CSV)
    if STATE["acc_gdf"] is not None and STATE["signature"] == sig:
        return STATE["acc_gdf"]
    for p in [CARACT_CSV, LIEUX_CSV, VEHICULES_CSV, USAGERS_CSV]:
        if not os.path.exists(p):
            raise RuntimeError(f"fichier manquant: {p}")

    caract = read_csv_cols(CARACT_CSV, ["Num_Acc", "lat", "long"])
    lieux = read_csv_cols(LIEUX_CSV, ["Num_Acc", "catr", "vma", "surf", "circ"])
    veh = read_csv_cols(VEHICULES_CSV, ["Num_Acc", "catv"])
    us = read_csv_cols(USAGERS_CSV, ["Num_Acc", "grav"])

    caract["lat"] = caract["lat"].map(_parse_fr_float)
    caract["long"] = caract["long"].map(_parse_fr_float)
    if "vma" in lieux.columns:
        lieux["vma"] = lieux["vma"].map(_parse_fr_float)
    us["grav"] = pd.to_numeric(us["grav"], errors="coerce")

    sev_map = {1: 0, 2: 12, 3: 4, 4: 1}
    us["sev_w"] = us["grav"].map(sev_map).fillna(0)
    us["injured_bool"] = us["grav"].isin([2, 3, 4])
    agg_u = us.groupby("Num_Acc", as_index=False).agg(
        sev_sum=("sev_w", "sum"), injured_any=("injured_bool", "any")
    )
    veh_counts = veh.groupby(["Num_Acc", "catv"]).size().reset_index(name="n")
    if veh_counts.empty:
        veh_pivot = pd.DataFrame({"Num_Acc": caract["Num_Acc"].unique()})
    else:
        veh_pivot = veh_counts.pivot_table(
            index="Num_Acc", columns="catv", values="n", fill_value=0
        )
        veh_pivot.columns = [f"veh_{c}" for c in veh_pivot.columns]
        veh_pivot = veh_pivot.reset_index()

    df = (
        caract.merge(agg_u, on="Num_Acc", how="left")
        .merge(lieux, on="Num_Acc", how="left")
        .merge(veh_pivot, on="Num_Acc", how="left")
    )
    df = df[(~df["lat"].isna()) & (~df["long"].isna())].copy()
    df["sev_sum"] = pd.to_numeric(df["sev_sum"], errors="coerce").fillna(0.0)
    if "injured_any" in df.columns:
        df["injured_any"] = (
            pd.Series(df["injured_any"], dtype="boolean").fillna(False).astype(bool)
        )
    else:
        df["injured_any"] = False
    for c in df.columns:
        if str(c).startswith("veh_"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["long"], df["lat"]), crs="EPSG:4326"
    )
    to2154 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    X, Y = to2154.transform(gdf.geometry.x.values, gdf.geometry.y.values)
    gdf["geometry"] = gpd.points_from_xy(X, Y, crs="EPSG:2154")

    if ctx is not None:
        gdf["lieux_mult"] = gdf.apply(
            lambda r: compute_lieux_multiplier(r, ctx), axis=1
        )
        veh_cols = [c for c in gdf.columns if str(c).startswith("veh_")]
        vmap = {}
        for _, r in gdf[["Num_Acc"] + veh_cols].iterrows():
            d = {}
            for c in veh_cols:
                try:
                    d[int(c.split("_")[-1])] = float(r[c])
                except:
                    pass
            vmap[str(r["Num_Acc"])] = d
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
    STATE["acc_gdf"] = gdf
    STATE["signature"] = sig
    return gdf


def split_line_fixed(line: LineString, L: float):
    res = []
    total = float(line.length)
    if total <= L:
        return [line]
    n = int(total // L)
    prev = 0.0
    for i in range(n):
        s, e = i * L, (i + 1) * L
        p, q = line.interpolate(s), line.interpolate(e)
        res.append(LineString([p.coords[0], q.coords[0]]))
        prev = e
    if total > prev:
        p, q = line.interpolate(prev), line.interpolate(total)
        res.append(LineString([p.coords[0], q.coords[0]]))
    return res


def segments_from_paths(lines: List[LineString], seg_len: float, crs=2154):
    segs = []
    for geom in lines:
        if isinstance(geom, LineString):
            segs += split_line_fixed(geom, seg_len)
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                segs += split_line_fixed(part, seg_len)
    if not segs:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs=crs)
    out = gpd.GeoDataFrame({"geometry": segs}, crs=crs)
    out["length_m"] = out.length
    return out


def _segments_intersection_multiplier(seg: gpd.GeoDataFrame):
    if seg.empty:
        seg["inter_mult"] = 1.0
        return seg
    ends = []
    for g in seg.geometry:
        c = list(g.coords)
        ends.append((round(c[0][0], 1), round(c[0][1], 1)))
        ends.append((round(c[-1][0], 1), round(c[-1][1], 1)))
    df = pd.DataFrame(ends, columns=["x", "y"])
    deg = df.value_counts().reset_index(name="deg")
    mapp = {(row.x, row.y): int(row.deg) for _, row in deg.iterrows()}
    d_left, d_right = [], []
    for g in seg.geometry:
        c = list(g.coords)
        a = (round(c[0][0], 1), round(c[0][1], 1))
        b = (round(c[-1][0], 1), round(c[-1][1], 1))
        d_left.append(mapp.get(a, 1))
        d_right.append(mapp.get(b, 1))
    seg["deg_max"] = np.maximum(d_left, d_right)
    seg["inter_mult"] = 1.0 + INTER_BETA * np.clip((seg["deg_max"] - 2) / 3.0, 0, 1.5)
    return seg


def join_accidents_to_segments_fast(
    acc_pts: gpd.GeoDataFrame, seg_gdf: gpd.GeoDataFrame, radius_m: float
):
    if seg_gdf.empty:
        return seg_gdf.assign(sev_sum=0.0, acc_count=0, rk_raw=0.0, rk_0_100=0.0)

    seg_cent = seg_gdf.geometry.interpolate(0.5, normalized=True)
    mx = np.array([p.x for p in seg_cent], dtype=float)
    my = np.array([p.y for p in seg_cent], dtype=float)
    tree = KDTree(np.c_[mx, my])

    pts = acc_pts[acc_pts["injured_any"] == True][
        ["Num_Acc", "sev_sum_adj", "geometry"]
    ].copy()
    if pts.empty:
        seg = seg_gdf.copy()
        seg["sev_sum"] = 0.0
        seg["acc_count"] = 0
        seg["length_km"] = seg["length_m"] / 1000.0
        mu = 0.5
        tau = float(EB_TAU_KM)
        seg["rk_raw"] = (seg["sev_sum"] + mu * tau) / (seg["length_km"] + tau)
        seg = _segments_intersection_multiplier(seg)
        seg["rk_raw"] = seg["rk_raw"] * seg["inter_mult"]
        rk = seg["rk_raw"].values
        rk = rk[rk > 0]
        q_ref = (
            float(np.quantile(rk, 0.90))
            if len(rk) >= 5
            else (float(rk.max()) if len(rk) else 1.0)
        )
        if not np.isfinite(q_ref) or q_ref <= 0:
            q_ref = 1.0
        seg["rk_0_100"] = (
            100.0 * np.log1p(seg["rk_raw"]) / np.log1p(1.5 * q_ref)
        ).clip(0, 100)
        return seg

    ax = np.array([p.x for p in pts.geometry], dtype=float)
    ay = np.array([p.y for p in pts.geometry], dtype=float)
    sev = np.array(
        pd.to_numeric(pts["sev_sum_adj"], errors="coerce").fillna(0.0), dtype=float
    )

    idxs = tree.query_radius(np.c_[ax, ay], r=radius_m, return_distance=False)
    sev_sum = np.zeros(len(seg_gdf), dtype=float)
    acc_cnt = np.zeros(len(seg_gdf), dtype=int)

    for i, neighbors in enumerate(idxs):
        if len(neighbors) == 0:
            continue
        sev_sum[neighbors] += sev[i]
        acc_cnt[neighbors] += 1

    seg = seg_gdf.copy()
    seg["sev_sum"] = sev_sum
    seg["acc_count"] = acc_cnt
    seg["length_km"] = seg["length_m"] / 1000.0

    S = seg["sev_sum"].astype(float)
    L = seg["length_km"].astype(float) + 1e-9
    mu = float(S.sum() / (L.sum() + 1e-9)) if S.sum() > 0 else 0.5
    tau = float(EB_TAU_KM)
    seg["rk_raw"] = (S + mu * tau) / (L + tau)

    seg = _segments_intersection_multiplier(seg)
    seg["rk_raw"] = seg["rk_raw"] * seg["inter_mult"]

    if SMOOTH_RADIUS_M > 0:
        t2 = KDTree(np.c_[mx, my])
        neigh = t2.query_radius(np.c_[mx, my], r=SMOOTH_RADIUS_M, return_distance=True)
        new = np.array(seg["rk_raw"].values, dtype=float)
        sigma = SMOOTH_RADIUS_M / 2.0
        for i, (ids, dist) in enumerate(zip(neigh[0], neigh[1])):
            if len(ids) <= 1:
                continue
            w = np.exp(-0.5 * (dist / (sigma + 1e-6)) ** 2)
            w = w / np.sum(w)
            m = float(np.sum(w * seg["rk_raw"].values[ids]))
            new[i] = SMOOTH_ALPHA * new[i] + (1.0 - SMOOTH_ALPHA) * m
        seg["rk_raw"] = new

    rk = seg["rk_raw"].values
    rk = rk[rk > 0]
    q_ref = (
        float(np.quantile(rk, 0.9))
        if len(rk) >= 5
        else (float(rk.max()) if len(rk) else 1.0)
    )
    if not np.isfinite(q_ref) or q_ref <= 0:
        q_ref = 1.0
    seg["rk_0_100"] = (100.0 * np.log1p(seg["rk_raw"]) / np.log1p(1.5 * q_ref)).clip(
        0, 100
    )
    return seg


def _iter_parts(geom):
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            if isinstance(g, LineString):
                yield g


def route_risk_index(route_geom, segs_df: gpd.GeoDataFrame) -> float:
    if segs_df.empty:
        return 0.0
    vals = []
    centers = segs_df.geometry.interpolate(0.5, normalized=True)
    mx = np.array([p.x for p in centers])
    my = np.array([p.y for p in centers])
    tree = KDTree(np.c_[mx, my])
    for part in _iter_parts(route_geom):
        L = float(part.length)
        if L <= 0:
            continue
        step = max(
            50.0,
            min(100.0, segs_df["length_m"].median() if "length_m" in segs_df else 75.0),
        )
        n = max(1, int(L // step))
        pts = [part.interpolate(i * step) for i in range(1, n + 1)]
        px = np.array([p.x for p in pts])
        py = np.array([p.y for p in pts])
        idx = tree.query(np.c_[px, py], k=1, return_distance=False).reshape(-1)
        vals.append(segs_df["rk_0_100"].values[idx])
    if not vals:
        return 0.0
    v = np.concatenate(vals)
    return float(np.mean(v)) if len(v) else 0.0


def exposure_stats(route_geom, segs_df: gpd.GeoDataFrame, step=75.0):
    below = 0.0
    above = 0.0
    acc = []
    if segs_df.empty:
        return 0.0, 0.0, 0.0
    centers = segs_df.geometry.interpolate(0.5, normalized=True)
    mx = np.array([p.x for p in centers])
    my = np.array([p.y for p in centers])
    tree = KDTree(np.c_[mx, my])
    for part in _iter_parts(route_geom):
        L = float(part.length)
        if L <= 0:
            continue
        n = max(1, int(L // step))
        pts = [part.interpolate(i * step) for i in range(1, n + 1)]
        px = np.array([p.x for p in pts])
        py = np.array([p.y for p in pts])
        idx = tree.query(np.c_[px, py], k=1, return_distance=False).reshape(-1)
        vals = segs_df["rk_0_100"].values[idx]
        below += float((vals < 50).sum()) * step
        above += float((vals >= 50).sum()) * step
        if len(vals):
            acc.append(vals)
    mean_v = float(np.mean(np.concatenate(acc))) if acc else 0.0
    return below, above, mean_v


def _headers(key: str):
    return {"Authorization": key, "Content-Type": "application/json"}


def _parse_ors_features(data):
    feats = data.get("features", [])
    out = []
    to2154 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    for f in feats:
        g = f.get("geometry", {})
        if g.get("type") != "LineString":
            continue
        coords = g.get("coordinates", []) or []
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
    return out


def _distinct(line: LineString, keep: List[LineString]) -> bool:
    if not keep:
        return True
    for l in keep:
        if line.hausdorff_distance(l) < 120:
            return False
    return True


async def _post_ors(client: httpx.AsyncClient, payload: dict, key: str):
    r = await client.post(ORS_URL, headers=_headers(key), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


async def ors_k_paths_fast(
    client: httpx.AsyncClient, o_lon, o_lat, d_lon, d_lat, want_k: int, key: str
):
    want_k = max(5, int(want_k))
    base = {"coordinates": [[o_lon, o_lat], [d_lon, d_lat]]}
    patterns = [
        {"share_factor": 0.85, "weight_factor": 1.6, "continue_straight": True},
        {"share_factor": 0.75, "weight_factor": 1.6, "continue_straight": False},
        {"share_factor": 0.65, "weight_factor": 1.8, "continue_straight": True},
    ]
    sem = asyncio.Semaphore(3)
    results, geoms = [], []

    async def one(cfg):
        payload = dict(base)
        payload["continue_straight"] = cfg["continue_straight"]
        payload["alternative_routes"] = {
            "target_count": 3,
            "share_factor": cfg["share_factor"],
            "weight_factor": cfg["weight_factor"],
        }
        async with sem:
            try:
                data = await _post_ors(client, payload, key)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    payload.pop("alternative_routes", None)
                    data = await _post_ors(client, payload, key)
                else:
                    return []
        return _parse_ors_features(data)

    tasks = [asyncio.create_task(one(cfg)) for cfg in patterns]
    try:
        for coro in asyncio.as_completed(tasks):
            batch = await coro
            for it in sorted(batch, key=lambda r: r["time_sec"]):
                g = it["geom2154"]
                if _distinct(g, geoms):
                    geoms.append(g)
                    results.append(it)
                    if len(results) >= want_k:
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        return results
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    global HTTP
    HTTP = httpx.AsyncClient(timeout=30)
    try:
        load_accidents(ctx=None)
    except Exception:
        pass
    yield
    if HTTP is not None:
        await HTTP.aclose()


app = FastAPI(
    title="Commute Risk API",
    version="6.1.1",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    status = {
        "ors_key_env": bool(ORS_API_KEY),
        "caract": os.path.exists(CARACT_CSV),
        "lieux": os.path.exists(LIEUX_CSV),
        "vehicules": os.path.exists(VEHICULES_CSV),
        "usagers": os.path.exists(USAGERS_CSV),
        "fast_mode": FAST_MODE,
        "join_radius_m": JOIN_RADIUS_M,
    }
    status["status"] = (
        "ok"
        if status["caract"]
        and status["lieux"]
        and status["vehicules"]
        and status["usagers"]
        else "missing_data"
    )
    return status


async def _ors_routes(
    client: httpx.AsyncClient, o_lon, o_lat, d_lon, d_lat, want_k: int, key: str
):
    return await ors_k_paths_fast(client, o_lon, o_lat, d_lon, d_lat, want_k, key)


@app.post("/risk/commute")
async def risk_commute(request: Request, body: CommuteBody):
    try:
        key = (
            request.headers.get("Authorization")
            or request.headers.get("X-ORS-Key")
            or ORS_API_KEY
        ).strip()
        if not key:
            return ORJSONResponse(
                status_code=400,
                content={
                    "error": "ORS_API_KEY manquant (env ORS_API_KEY ou header Authorization/X-ORS-Key)"
                },
            )

        sig = _sig(body)
        tok = RESULT_CACHE.get(sig)
        if tok and tok in STORE:
            return {
                "token": tok,
                "segments_geojson_url": f"/segments/{tok}",
                "route_aller_geojson_url": f"/route/aller/{tok}",
                "route_retour_geojson_url": f"/route/retour/{tok}",
                "alts_aller_geojson_url": f"/alts/aller/{tok}",
                "alts_retour_geojson_url": f"/alts/retour/{tok}",
                "map_url": f"/map/{tok}",
                "metrics_url": f"/metrics/{tok}",
            }

        acc = load_accidents(body.context)
        client = HTTP or httpx.AsyncClient(timeout=30)
        aller, retour = await asyncio.gather(
            _ors_routes(
                client,
                body.orig_lon,
                body.orig_lat,
                body.dest_lon,
                body.dest_lat,
                max(5, body.k_paths),
                key,
            ),
            _ors_routes(
                client,
                body.dest_lon,
                body.dest_lat,
                body.orig_lon,
                body.orig_lat,
                max(5, body.k_paths),
                key,
            ),
        )
        if not aller or not retour:
            return ORJSONResponse(
                status_code=502, content={"error": "ORS n'a renvoyé aucun itinéraire"}
            )

        def filter_by_time(routes, tcap):
            if tcap is None:
                return routes
            lo, hi = 0.7 * tcap, 1.3 * tcap
            out = [r for r in routes if lo <= r["time_sec"] <= hi]
            return out or routes

        t_target = float(body.hour) * 60.0 if body.hour and body.hour > 0 else None
        aller = filter_by_time(aller, t_target)
        retour = filter_by_time(retour, t_target)

        tmin_a = float(min(r["time_sec"] for r in aller))
        tmin_r = float(min(r["time_sec"] for r in retour))
        aller = [r for r in aller if r["time_sec"] <= body.time_cap_ratio * tmin_a] or [
            min(aller, key=lambda x: x["time_sec"])
        ]
        retour = [
            r for r in retour if r["time_sec"] <= body.time_cap_ratio * tmin_r
        ] or [min(retour, key=lambda x: x["time_sec"])]

        probs_a = np.array(
            [(1.0 - body.alt_geom_p) ** i * body.alt_geom_p for i in range(len(aller))],
            dtype=float,
        )
        probs_a /= probs_a.sum()
        probs_r = np.array(
            [
                (1.0 - body.alt_geom_p) ** i * body.alt_geom_p
                for i in range(len(retour))
            ],
            dtype=float,
        )
        probs_r /= probs_r.sum()
        alts_a = [
            (r["geom2154"], float(r["time_sec"]), float(p), 0.0)
            for r, p in zip(sorted(aller, key=lambda x: x["time_sec"]), probs_a)
        ]
        alts_r = [
            (r["geom2154"], float(r["time_sec"]), float(p), 0.0)
            for r, p in zip(sorted(retour, key=lambda x: x["time_sec"]), probs_r)
        ]
        base_a = alts_a[0][0]
        base_r = alts_r[0][0]

        lines_all = [g for g, _, _, _ in alts_a] + [g for g, _, _, _ in alts_r]
        acc_corr = acc[
            acc.intersects(unary_union(lines_all).buffer(body.corridor_m))
        ].copy()
        seg_box = segments_from_paths(lines_all, body.seg_len_m, crs=2154)
        if not seg_box.empty:
            seg_box = join_accidents_to_segments_fast(acc_corr, seg_box, JOIN_RADIUS_M)

        def score_alts(alts, seg):
            out = []
            for g, T, p, _ in alts:
                rk = route_risk_index(g, seg) if not seg.empty else 0.0
                out.append((g, T, p, rk))
            return out

        alts_a = score_alts(alts_a, seg_box)
        alts_r = score_alts(alts_r, seg_box)

        exp_risk_a = float(sum(p * rk for (_g, T, p, rk) in alts_a))
        exp_risk_r = float(sum(p * rk for (_g, T, p, rk) in alts_r))
        exp_time_a_min = float(sum(p * (T / 60.0) for (_g, T, p, _rk) in alts_a))
        exp_time_r_min = float(sum(p * (T / 60.0) for (_g, T, p, _rk) in alts_r))
        exp_roundtrip_min = exp_time_a_min + exp_time_r_min

        below_a, above_a, mean_a = (
            (0.0, 0.0, 0.0)
            if seg_box.empty
            else exposure_stats(base_a, seg_box, step=max(60.0, body.seg_len_m))
        )
        below_r, above_r, mean_r = (
            (0.0, 0.0, 0.0)
            if seg_box.empty
            else exposure_stats(base_r, seg_box, step=max(60.0, body.seg_len_m))
        )

        corridor_ref = (
            float(np.median(seg_box["rk_0_100"]))
            if (seg_box is not None and len(seg_box))
            else 10.0
        )
        ref_median = RISK_REF_MEDIAN if (RISK_REF_MEDIAN is not None) else corridor_ref
        combined_expected = (exp_risk_a + exp_risk_r) / 2.0
        annual_minutes_user = exp_roundtrip_min * (body.trips_per_year / 2.0)
        scale_time = annual_minutes_user / (MINUTES_ANNUELLES_REF + 1e-9)
        annualized = 50.0 * (combined_expected / max(1e-6, ref_median)) * scale_time

        token = uuid.uuid4().hex
        tmpdir = tempfile.gettempdir()
        seg_path = os.path.join(tmpdir, f"segments_{token}.geojson")
        alts_a_path = os.path.join(tmpdir, f"alts_aller_{token}.geojson")
        alts_r_path = os.path.join(tmpdir, f"alts_retour_{token}.geojson")
        base_a_path = os.path.join(tmpdir, f"route_aller_{token}.geojson")
        base_r_path = os.path.join(tmpdir, f"route_retour_{token}.geojson")

        if not seg_box.empty:
            seg_box.to_crs(4326).to_file(seg_path, driver="GeoJSON")
        gpd.GeoDataFrame(geometry=[base_a], crs=2154).to_crs(4326).to_file(
            base_a_path, driver="GeoJSON"
        )
        gpd.GeoDataFrame(geometry=[base_r], crs=2154).to_crs(4326).to_file(
            base_r_path, driver="GeoJSON"
        )
        gpd.GeoDataFrame(geometry=[g for g, _, _, _ in alts_a], crs=2154).to_crs(
            4326
        ).to_file(alts_a_path, driver="GeoJSON")
        gpd.GeoDataFrame(geometry=[g for g, _, _, _ in alts_r], crs=2154).to_crs(
            4326
        ).to_file(alts_r_path, driver="GeoJSON")

        STORE[token] = {
            "segments_path": seg_path if not seg_box.empty else None,
            "base_a_path": base_a_path,
            "base_r_path": base_r_path,
            "alts_a_path": alts_a_path,
            "alts_r_path": alts_r_path,
            "acc_corr": acc_corr,
            "alts_a": alts_a,
            "alts_r": alts_r,
            "orig": (body.orig_lon, body.orig_lat),
            "dest": (body.dest_lon, body.dest_lat),
            "metrics": {
                "calibration": {
                    "ref_daily_commute_min": REF_DAILY_COMMUTE_MIN,
                    "trips_per_year_ref": TRIPS_PER_YEAR_REF,
                    "risk_ref_median_effective": ref_median,
                    "minutes_annuelles_ref": MINUTES_ANNUELLES_REF,
                    "annual_minutes_user": annual_minutes_user,
                    "scale_time": scale_time,
                },
                "params": {
                    "trips_per_year": body.trips_per_year,
                    "seg_len_m": body.seg_len_m,
                    "corridor_m": body.corridor_m,
                    "time_cap_ratio": body.time_cap_ratio,
                    "k_paths_min_per_direction": max(5, body.k_paths),
                    "alt_geom_p": body.alt_geom_p,
                    "duration_target_min": body.hour,
                    "eb_tau_km": EB_TAU_KM,
                    "smooth_radius_m": SMOOTH_RADIUS_M,
                    "smooth_alpha": SMOOTH_ALPHA,
                    "inter_beta": INTER_BETA,
                    "join_radius_m": JOIN_RADIUS_M,
                    "fast_mode": FAST_MODE,
                },
                "directions": {
                    "aller": {
                        "baseline_time_sec": float(min(r["time_sec"] for r in aller)),
                        "expected_route_risk_index_0_100": float(exp_risk_a),
                        "expected_time_min": float(exp_time_a_min),
                        "exposure": {
                            "below_50_km": below_a / 1000.0,
                            "above_50_km": above_a / 1000.0,
                            "mean_risk_along_best": mean_a,
                        },
                        "alternatives": [
                            {
                                "prob": float(p),
                                "time_sec": float(T),
                                "risk_index_0_100": float(rk),
                            }
                            for (_g, T, p, rk) in sorted(
                                alts_a, key=lambda x: x[2], reverse=True
                            )
                        ],
                    },
                    "retour": {
                        "baseline_time_sec": float(min(r["time_sec"] for r in retour)),
                        "expected_route_risk_index_0_100": float(exp_risk_r),
                        "expected_time_min": float(exp_time_r_min),
                        "exposure": {
                            "below_50_km": below_r / 1000.0,
                            "above_50_km": above_r / 1000.0,
                            "mean_risk_along_best": mean_r,
                        },
                        "alternatives": [
                            {
                                "prob": float(p),
                                "time_sec": float(T),
                                "risk_index_0_100": float(rk),
                            }
                            for (_g, T, p, rk) in sorted(
                                alts_r, key=lambda x: x[2], reverse=True
                            )
                        ],
                    },
                },
                "combined": {
                    "expected_route_risk_index_0_100": float(
                        (exp_risk_a + exp_risk_r) / 2.0
                    ),
                    "expected_roundtrip_time_min": float(exp_roundtrip_min),
                    "annualized_risk_index_0_100": float(annualized),
                },
            },
        }
        RESULT_CACHE[sig] = token
        return {
            "token": token,
            "segments_geojson_url": f"/segments/{token}",
            "route_aller_geojson_url": f"/route/aller/{token}",
            "route_retour_geojson_url": f"/route/retour/{token}",
            "alts_aller_geojson_url": f"/alts/aller/{token}",
            "alts_retour_geojson_url": f"/alts/retour/{token}",
            "map_url": f"/map/{token}",
            "metrics_url": f"/metrics/{token}",
        }
    except httpx.HTTPStatusError as e:
        return ORJSONResponse(
            status_code=e.response.status_code,
            content={
                "error": f"ORS HTTP {e.response.status_code}: {e.response.text[:300]}"
            },
        )
    except Exception as e:
        return ORJSONResponse(
            status_code=500, content={"error": f"Erreur: {type(e).__name__}: {e}"}
        )


@app.get("/segments/{token}")
def get_segments(token: str):
    if token not in STORE:
        return ORJSONResponse(status_code=404, content={"error": "inconnu"})
    p = STORE[token]["segments_path"]
    if not p or not os.path.exists(p):
        return ORJSONResponse(status_code=404, content={"error": "introuvable"})
    with open(p, "r", encoding="utf-8") as f:
        return ORJSONResponse(json.load(f))


@app.get("/route/aller/{token}")
def get_route_aller(token: str):
    if token not in STORE:
        return ORJSONResponse(status_code=404, content={"error": "inconnu"})
    p = STORE[token]["base_a_path"]
    if not os.path.exists(p):
        return ORJSONResponse(status_code=404, content={"error": "introuvable"})
    with open(p, "r", encoding="utf-8") as f:
        return ORJSONResponse(json.load(f))


@app.get("/route/retour/{token}")
def get_route_retour(token: str):
    if token not in STORE:
        return ORJSONResponse(status_code=404, content={"error": "inconnu"})
    p = STORE[token]["base_r_path"]
    if not os.path.exists(p):
        return ORJSONResponse(status_code=404, content={"error": "introuvable"})
    with open(p, "r", encoding="utf-8") as f:
        return ORJSONResponse(json.load(f))


@app.get("/alts/aller/{token}")
def get_alts_aller(token: str):
    if token not in STORE:
        return ORJSONResponse(status_code=404, content={"error": "inconnu"})
    p = STORE[token]["alts_a_path"]
    if not p or not os.path.exists(p):
        return ORJSONResponse(status_code=404, content={"error": "introuvable"})
    with open(p, "r", encoding="utf-8") as f:
        return ORJSONResponse(json.load(f))


@app.get("/alts/retour/{token}")
def get_alts_retour(token: str):
    if token not in STORE:
        return ORJSONResponse(status_code=404, content={"error": "inconnu"})
    p = STORE[token]["alts_r_path"]
    if not p or not os.path.exists(p):
        return ORJSONResponse(status_code=404, content={"error": "introuvable"})
    with open(p, "r", encoding="utf-8") as f:
        return ORJSONResponse(json.load(f))


def make_map(
    origin_wgs84,
    dest_wgs84,
    base_aller,
    alts_aller,
    base_retour,
    alts_retour,
    seg_gdf,
    acc_corr,
):
    m = folium.Map(
        location=[origin_wgs84[1], origin_wgs84[0]],
        zoom_start=11,
        tiles="cartodbpositron",
    )
    folium.map.CustomPane("risk", z_index=640).add_to(m)
    folium.map.CustomPane("routes", z_index=720).add_to(m)
    folium.map.CustomPane("accs", z_index=740).add_to(m)

    qbreaks = [20, 40, 60, 80]
    if seg_gdf is not None and not seg_gdf.empty and "rk_0_100" in seg_gdf.columns:
        vals = seg_gdf["rk_0_100"].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(vals) >= 10:
            qbreaks = list(np.quantile(vals, [0.2, 0.4, 0.6, 0.8]))
    colors = ["#2ECC71", "#9BE7C2", "#F9E79F", "#F5B041", "#C0392B"]

    def color_for(v):
        if v <= qbreaks[0]:
            return colors[0]
        if v <= qbreaks[1]:
            return colors[1]
        if v <= qbreaks[2]:
            return colors[2]
        if v <= qbreaks[3]:
            return colors[3]
        return colors[4]

    if seg_gdf is not None and not seg_gdf.empty:
        segs_wgs = seg_gdf.to_crs(4326)
        folium.GeoJson(
            segs_wgs[
                ["rk_0_100", "acc_count", "sev_sum", "geometry"]
            ].__geo_interface__,
            pane="risk",
            style_function=lambda f: {
                "color": color_for(f["properties"]["rk_0_100"]),
                "weight": 5,
                "opacity": 0.9,
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=["rk_0_100", "acc_count", "sev_sum"],
                aliases=["Score", "Accidents", "Sévérité"],
                localize=True,
            ),
            name="Risque 50–100 m",
        ).add_to(m)

    route_colors = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
        "#1f78b4",
        "#33a02c",
    ]
    thin_main = 3.0
    thick_alt = 6.0

    if alts_aller:
        for i, (g, T, p, r) in enumerate(
            sorted(alts_aller, key=lambda x: x[2], reverse=True)
        ):
            geo = gpd.GeoSeries([g], crs=2154).to_crs(4326).iloc[0]
            col = route_colors[i % len(route_colors)]
            folium.PolyLine(
                [(y, x) for x, y in geo.coords],
                weight=thick_alt,
                color=col,
                opacity=0.98,
                tooltip=f"Aller alt p={p:.1%} • {T/60:.1f} min • risque {r:.1f}",
                pane="routes",
            ).add_to(m)
    if base_aller is not None:
        geo = gpd.GeoSeries([base_aller], crs=2154).to_crs(4326).iloc[0]
        col = route_colors[len(alts_aller) % len(route_colors)]
        folium.PolyLine(
            [(y, x) for x, y in geo.coords],
            weight=thin_main,
            color=col,
            opacity=1.0,
            tooltip="Aller principal",
            pane="routes",
        ).add_to(m)

    if alts_retour:
        for i, (g, T, p, r) in enumerate(
            sorted(alts_retour, key=lambda x: x[2], reverse=True)
        ):
            geo = gpd.GeoSeries([g], crs=2154).to_crs(4326).iloc[0]
            col = route_colors[i % len(route_colors)]
            folium.PolyLine(
                [(y, x) for x, y in geo.coords],
                weight=thick_alt,
                color=col,
                opacity=0.98,
                tooltip=f"Retour alt p={p:.1%} • {T/60:.1f} min • risque {r:.1f}",
                pane="routes",
            ).add_to(m)
    if base_retour is not None:
        geo = gpd.GeoSeries([base_retour], crs=2154).to_crs(4326).iloc[0]
        col = route_colors[(len(alts_retour)) % len(route_colors)]
        folium.PolyLine(
            [(y, x) for x, y in geo.coords],
            weight=thin_main,
            color=col,
            opacity=1.0,
            tooltip="Retour principal",
            pane="routes",
        ).add_to(m)

    folium.Marker(
        [origin_wgs84[1], origin_wgs84[0]],
        icon=folium.Icon(color="green"),
        tooltip="Origine",
        pane="routes",
    ).add_to(m)
    folium.Marker(
        [dest_wgs84[1], dest_wgs84[0]],
        icon=folium.Icon(color="red"),
        tooltip="Destination",
        pane="routes",
    ).add_to(m)

    try:
        hot = (
            acc_corr.to_crs(4326).sort_values("sev_sum_adj", ascending=False).head(300)
        )
        for _, r in hot.iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x],
                radius=3,
                color="#8E44AD",
                fill=True,
                fill_opacity=0.6,
                pane="accs",
            ).add_to(m)
    except Exception:
        pass

    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


@app.get("/map/{token}")
def get_map(token: str):
    if token not in STORE:
        return HTMLResponse("<b>inconnu</b>", status_code=404)
    base_a = gpd.read_file(STORE[token]["base_a_path"]).to_crs(2154).iloc[0].geometry
    base_r = gpd.read_file(STORE[token]["base_r_path"]).to_crs(2154).iloc[0].geometry
    seg_gdf = (
        gpd.read_file(STORE[token]["segments_path"]).to_crs(2154)
        if STORE[token]["segments_path"]
        else gpd.GeoDataFrame(columns=["geometry"], geometry=[])
    )
    acc_corr = STORE[token]["acc_corr"]
    origin_wgs84 = STORE[token]["orig"]
    dest_wgs84 = STORE[token]["dest"]
    alts_a = STORE[token]["alts_a"]
    alts_r = STORE[token]["alts_r"]
    html = make_map(
        origin_wgs84, dest_wgs84, base_a, alts_a, base_r, alts_r, seg_gdf, acc_corr
    )
    return HTMLResponse(html)


@app.get("/metrics/{token}")
def get_metrics(token: str):
    if token not in STORE:
        return ORJSONResponse(status_code=404, content={"error": "inconnu"})
    return ORJSONResponse(STORE[token]["metrics"])
