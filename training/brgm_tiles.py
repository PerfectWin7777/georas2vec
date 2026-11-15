# Vraies tuiles BRGM 1/50 000 : scan (WMS SCAN_D_GEOL50) + vérité BD Charm-50 (formations, failles).
#   python brgm_tiles.py 029 021 ...          -> ../brgm/tiles/<dep>.parquet (split train)
#   python brgm_tiles.py --test 014           -> ../brgm/tiles/014_test.parquet (Calvados : test uniquement)
# Une instance = composante connexe d'une même notation ; unité minimale 40 px ; failles = lignes dont DESCR contient « faille »
# (couches L_STRUCT et L_FGEOL). Contrôle d'alignement : part des limites vectorielles qui tombent sur un contraste du scan.
import os, sys, io, glob, json, random, time, numpy as np, pandas as pd, geopandas as gpd, urllib.request
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_eval as SE
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "brgm")
S, PX, T = 8600, 2048, 512
WMS = "http://geoservices.brgm.fr/geologie?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=SCAN_D_GEOL50&STYLES=&CRS=EPSG:2154&BBOX={},{},{},{}&WIDTH=2048&HEIGHT=2048&FORMAT=image/png"


def layers(dep):
    root = f"{B}/dl/d{dep}" if dep != "014" else f"{B}/dl14"
    P = gpd.read_file(glob.glob(f"{root}/**/*S_FGEOL*.shp", recursive=True)[0])
    L = pd.concat([gpd.read_file(f) for f in glob.glob(f"{root}/**/*L_STRUCT*.shp", recursive=True) + glob.glob(f"{root}/**/*L_FGEOL*.shp", recursive=True)])
    L = L[L.DESCR.str.contains("aille", case=False, na=False)]
    return P, L


def windows(P, n, rng, min_units=15):
    b = P.total_bounds; cand = []
    sidx = P.sindex
    for _ in range(4000):
        x, y = rng.uniform(b[0], b[2] - S), rng.uniform(b[1], b[3] - S)
        hit = P.iloc[sidx.query(gpd.GeoSeries.from_xy([x + S / 2], [y + S / 2]).buffer(S / 2, cap_style=3).iloc[0])]
        Q = hit.clip((x, y, x + S, y + S))
        if Q.area.sum() / S ** 2 > 0.98 and Q.NOTATION.nunique() >= min_units // 3 and len(Q) >= min_units:
            if all(abs(x - a) > S or abs(y - c) > S for a, c in cand):
                cand.append((x, y))
        if len(cand) >= n:
            break
    return cand


def fetch(x, y):
    for k in range(4):
        try:
            with urllib.request.urlopen(WMS.format(int(x), int(y), int(x) + S, int(y) + S), timeout=300) as r:
                return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
        except Exception as e:
            time.sleep(5 * (k + 1))
    return None


def labels(P, L, x, y):
    tr = from_bounds(x, y, x + S, y + S, PX, PX)
    Q = P.cx[x:x + S, y:y + S]
    codes = {c: k + 1 for k, c in enumerate(sorted(Q.NOTATION.fillna("?").unique()))}
    unit = rasterize(zip(Q.geometry, Q.NOTATION.fillna("?").map(codes)), out_shape=(PX, PX), transform=tr, fill=0, dtype="int32")
    inst = np.zeros_like(unit); n = 0
    for c in np.unique(unit[unit > 0]):
        lc, k = ndi.label(unit == c); inst[lc > 0] = lc[lc > 0] + n; n += k
    inst = SE.remove_small(inst, 40) if inst.max() else inst
    flt = rasterize(((g, 1) for g in L.cx[x:x + S, y:y + S].geometry), out_shape=(PX, PX), transform=tr, fill=0, dtype="uint8")
    names = {v: k for k, v in codes.items()}
    unit_of = {int(i): names.get(int(np.bincount(unit[inst == i]).argmax()), "?") for i in np.unique(inst[inst > 0])}
    return inst, flt, unit_of


def alignment(rgb, inst):
    """Part des pixels de limite vectorielle à <= 2 px d'un fort contraste de couleur dans le scan."""
    lab = rgb2lab(rgb); g = sum(np.hypot(ndi.sobel(lab[..., c], 0), ndi.sobel(lab[..., c], 1)) for c in (1, 2)) + 0.5 * np.hypot(ndi.sobel(lab[..., 0], 0), ndi.sobel(lab[..., 0], 1))
    strong = ndi.binary_dilation(g > np.percentile(g, 85), iterations=2)
    e = np.zeros(inst.shape, bool); e[:, :-1] |= inst[:, :-1] != inst[:, 1:]; e[:-1, :] |= inst[:-1, :] != inst[1:, :]
    return float(strong[e].mean()) if e.any() else 0.0


def png(a, mode=None):
    bio = io.BytesIO(); Image.fromarray(a, mode).save(bio, "PNG"); return {"bytes": bio.getvalue(), "path": None}


def build(dep, n_win, split, rng, min_align=0.55):
    P, L = layers(dep); rows, qc = [], []
    for (x, y) in windows(P, n_win, rng):
        rgb = fetch(x, y)
        if rgb is None:
            continue
        inst, flt, unit_of = labels(P, L, x, y); al = alignment(rgb, inst); qc.append(dict(dep=dep, x=int(x), y=int(y), alignement=al, unites=int(inst.max())))
        print(dep, int(x), int(y), "alignement %.2f" % al, "unités", inst.max(), flush=True)
        if al < min_align:
            continue
        for i in range(0, PX, T):
            for j in range(0, PX, T):
                ti = inst[i:i + T, j:j + T]; ids = np.unique(ti[ti > 0])
                if len(ids) < 2:
                    continue
                mp = np.zeros(ti.max() + 1, np.int32); mp[ids] = np.arange(1, len(ids) + 1); t2 = mp[ti].astype(np.uint16)
                units = {str(k + 1): {"unit": unit_of[int(v)], "age": "", "lithology": ""} for k, v in enumerate(ids)}
                rows.append(dict(id=f"BRGM-{dep}_{int(x)}_{int(y)}_{i}_{j}", source="BRGM-50k", continent="Europe", kind="real", split=split,
                                 image=png(np.ascontiguousarray(rgb[i:i + T, j:j + T])), instances=png(t2, "I;16"),
                                 faults=png((flt[i:i + T, j:j + T] > 0).astype(np.uint8) * 255), n_instances=len(ids),
                                 fault_pixels=int(flt[i:i + T, j:j + T].sum()), px_size_m=S / PX, crs="EPSG:2154",
                                 bounds=json.dumps([x + j * S / PX, y + S - (i + T) * S / PX, x + (j + T) * S / PX, y + S - i * S / PX]),
                                 units=json.dumps(units), render="{}", texts="[]"))
    return rows, qc


if __name__ == "__main__":
    os.makedirs(f"{B}/tiles", exist_ok=True)
    test = sys.argv[1] == "--test"; deps = sys.argv[2:] if test else sys.argv[1:]
    for dep in deps:
        out = f"{B}/tiles/{dep}{'_test' if test else ''}.parquet"
        if os.path.exists(out):
            continue
        rows, qc = build(dep, 4 if test else 6, "test" if test else "train", random.Random(int(dep)))
        pd.DataFrame(qc).to_csv(out.replace(".parquet", "_qc.csv"), index=False)
        if rows:
            pd.DataFrame(rows).to_parquet(out, index=False)
        print(dep, len(rows), "tuiles ->", out, flush=True)
