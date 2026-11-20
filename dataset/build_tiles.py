# =====================================================================================
#  GEORAS2VEC : sources vectorielles -> tuiles (scan réaliste, instances, failles)
#  - tuiles RÉELLES  : scan GMNA + vecteurs USGS alignés
#  - tuiles SYNTHÉTIQUES : la carte est "imprimée" à partir des polygones, dans le style
#    d'une vraie carte géologique (couleurs chronostratigraphiques, motifs, contacts,
#    failles à dents, étiquettes, mer, graticule, rivières...) puis "scannée".
#  Vérité terrain commune :
#    inst   : uint16, 0 = pas de polygone (mer / hors carte), 1..N = composantes connexes
#             d'une MÊME unité (deux polygones voisins de la même unité = une seule instance,
#             car aucune limite n'est visible entre eux sur la carte)
#    faults : uint8 {0,1}, tracé des failles (1 px)
# =====================================================================================
import io, json, math, re, zlib, colorsys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
from shapely.geometry import box
from shapely.validation import make_valid
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from skimage.measure import label as cc_label, find_contours
from PIL import Image, ImageDraw, ImageFont

TILE, SS = 512, 2          # tuile finale 512 px, rendu interne sur-échantillonné x2 (anti-crénelage)
R = TILE * SS

# ------------------------------------------------------------------ polices
_FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONTS = {
    "sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "ital": "DejaVuSans-Oblique.ttf",
    "bital": "DejaVuSans-BoldOblique.ttf", "serif": "DejaVuSerif.ttf", "serifi": "DejaVuSerif-Italic.ttf",
    "serifb": "DejaVuSerif-Bold.ttf", "cond": "DejaVuSansMono.ttf",
}


@lru_cache(maxsize=256)
def font(name, size):
    try:
        return ImageFont.truetype(str(_FD / FONTS[name]), int(size))
    except Exception:
        return ImageFont.load_default()


def _seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


# ------------------------------------------------------------------ couleurs (charte chronostratigraphique CGMW/ICS)
AGE_COLORS = [
    (r"holocene|recent", (254, 242, 224)), (r"pleistocene", (255, 242, 174)), (r"quaternary|kvart", (249, 249, 127)),
    (r"pliocene", (255, 255, 153)), (r"miocene", (255, 255, 0)), (r"neogene", (255, 230, 25)),
    (r"oligocene", (254, 192, 122)), (r"pal(a)?eocene", (253, 167, 95)), (r"eocene", (253, 180, 108)),
    (r"pal(a)?eogene", (253, 154, 82)), (r"tertiary", (255, 200, 60)), (r"c(a)?enozoic", (242, 249, 29)),
    (r"cretaceous", (127, 198, 78)), (r"jurassic", (52, 178, 201)), (r"triassic", (129, 43, 146)),
    (r"mesozoic", (103, 197, 202)), (r"permian", (240, 64, 40)),
    (r"carboniferous|pennsylvanian|mississippian", (103, 165, 153)), (r"devonian", (203, 140, 55)),
    (r"silurian", (179, 225, 182)), (r"ordovician", (0, 146, 112)), (r"cambrian", (127, 160, 86)),
    (r"pal(a)?eozoic", (153, 192, 141)),
    (r"ediacaran|cryogenian|tonian|neoproterozoic|late proterozoic", (254, 179, 66)),
    (r"mesoproterozoic|middle proterozoic", (253, 180, 98)),
    (r"pal(a)?eoproterozoic|early proterozoic", (247, 67, 112)), (r"proterozoic", (247, 53, 99)),
    (r"archean|archaean", (240, 4, 127)), (r"precambrian", (247, 67, 112)),
]
LITHO_COLORS = [  # quand l'âge est inconnu (ex. GEO-LiM) : carte lithologique
    (r"water|lake", (190, 225, 245)), (r"\bice\b|glacier", (236, 246, 252)), (r"peat", (170, 140, 100)),
    (r"gypsum|evaporite", (250, 200, 230)), (r"pure carbonate", (140, 195, 235)),
    (r"carbonate|limestone|dolom", (175, 215, 225)), (r"sandstone", (250, 222, 150)),
    (r"claystone|shale|mudstone|clay", (215, 195, 160)), (r"acid|granit|felsic", (240, 120, 140)),
    (r"intermediate", (235, 160, 180)), (r"mafic|basalt", (90, 150, 100)),
    (r"metamorph|schist|gneiss", (190, 160, 215)), (r"alluvi|sediment", (255, 240, 170)),
]
PLUTONIC = re.compile(r"plutonic|intrusive|granit|tonalit|diorit|gabbro|syenit|monzon|batholith|granodior|pluton", re.I)
VOLCANIC = re.compile(r"volcanic|basalt|andesit|rhyolit|tuff|lava|ignimbr|trachy|pyroclast", re.I)
METAMORPH = re.compile(r"metamorph|schist|gneiss|crystalline|quartzite|marble|migmat|amphibolit|slate|phyllit", re.I)
UNCONSOL = re.compile(r"alluvi|unconsolidated|sand dune|colluv|aeolian|eolian|surficial|regolith|playa|dune", re.I)


def _match(table, text):
    t = (text or "").lower()
    for pat, rgb in table:
        if re.search(pat, t):
            return np.array(rgb, np.float32)
    return None


def _mix(a, b, w):
    return a * (1 - w) + np.asarray(b, np.float32) * w


def base_color(age, lith):
    """Couleur de légende 'officielle' d'une unité : âge (charte CGMW), modulée par la lithologie."""
    age = age if isinstance(age, str) else ""
    lith = lith if isinstance(lith, str) else ""
    c = _match(AGE_COLORS, age)
    if c is None:
        c = _match(LITHO_COLORS, lith)
    if c is None:
        h = (_seed(age, lith) % 1000) / 1000
        c = np.array(colorsys.hsv_to_rgb(h, 0.45, 0.9), np.float32) * 255
    if re.search(r"\bice\b|glacier", lith, re.I):
        return np.array((236, 246, 252), np.float32)
    if PLUTONIC.search(lith):
        c = _mix(c, (232, 50, 110), 0.5)
    elif METAMORPH.search(lith):
        c = _mix(c, (165, 120, 185), 0.35)
    return c


def pattern_pref(lith):
    lith = lith if isinstance(lith, str) else ""
    if VOLCANIC.search(lith):
        return "v"
    if UNCONSOL.search(lith):
        return "dots"
    if PLUTONIC.search(lith):
        return "cross"
    return None


# =====================================================================================
#  SOURCES
# =====================================================================================
class Source:
    """Unités (polygones + uid), table des unités, failles, CRS métrique, plage d'échelles."""

    def __init__(self, name, continent, units, tab, faults, px_range, min_land=0.55):
        self.name, self.continent = name, continent
        self.units, self.tab, self.faults = units, tab, faults
        self.px_range, self.min_land = px_range, min_land
        self.crs = units.crs
        _ = units.sindex
        if faults is not None and len(faults):
            _ = faults.sindex
        self.lut_rgb = np.zeros((int(tab.index.max()) + 1, 3), np.float32)
        for uid, r in tab.iterrows():
            self.lut_rgb[uid] = r.rgb
        self.codes = tab["code"].to_dict()
        self.pref = tab["pref"].to_dict()

    def unit_meta(self, uid):
        r = self.tab.loc[uid]
        return {"unit": str(r.code), "age": str(r.age or ""), "lithology": str(r.lith or "")[:120]}


def _clean(g, crs, simplify_m=0.0):
    g = g[g.geometry.notnull() & ~g.geometry.is_empty].to_crs(crs)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    if simplify_m:
        g["geometry"] = g.geometry.simplify(simplify_m, preserve_topology=True)
    g = g[~g.geometry.is_empty & g.geom_type.isin(["Polygon", "MultiPolygon", "GeometryCollection"])]
    return g.reset_index(drop=True)


def _read(path, **kw):
    for mode in ("fix", "ignore"):
        try:
            return gpd.read_file(path, on_invalid=mode, **kw)
        except Exception:
            pass
    return gpd.read_file(path, **kw)


def build_units(g, key, code, age, lith, order=None):
    """Attribue un uid par unité (clé) et construit la table des unités."""
    g = g.copy()
    g["_key"] = g[key].astype(str).fillna("?")
    keys = pd.Index(sorted(g["_key"].unique()))
    g["uid"] = keys.get_indexer(g["_key"]) + 1
    g["order"] = order if order is not None else 0
    first = g.drop_duplicates("uid").set_index("uid").sort_index()
    tab = pd.DataFrame(index=first.index)
    tab["code"] = (first[code] if code in first else first["_key"]).fillna("").astype(str).str.slice(0, 10)
    tab["age"] = (first[age] if isinstance(age, str) else age(first)).fillna("").astype(str)
    tab["lith"] = (first[lith] if isinstance(lith, str) else lith(first)).fillna("").astype(str)
    tab["rgb"] = [base_color(a, l) for a, l in zip(tab.age, tab.lith)]
    tab["pref"] = [pattern_pref(l) for l in tab.lith]
    return g[["uid", "order", "geometry"]], tab


def load_gmna(units_shp, faults_shp, crs):
    u = _clean(_read(units_shp), crs)
    agef = lambda d: d["MIN_AGE"].fillna("").astype(str) + " / " + d["MAX_AGE"].fillna("").astype(str)
    lithf = lambda d: d["ROCKTYPE"].fillna("").astype(str) + " " + d["LITHOLOGY"].fillna("").astype(str)
    units, tab = build_units(u, "UNIT_ABBRE", "UNIT_ABBRE", agef, lithf)
    f = _read(faults_shp).to_crs(crs)
    f = f[f.geometry.notnull()].reset_index(drop=True)
    f["kind"] = "generic"
    f["concealed"] = False
    return Source("GMNA", "North America", units, tab, f[["kind", "concealed", "geometry"]], (300, 600))


def load_geolim(gpkg, simplify_m=40):
    g = _clean(gpd.read_file(gpkg), "EPSG:3035", simplify_m)
    abbr = {"mix carbonate rocks": "Cm", "sandstone": "Ss", "claystone": "Cl", "pure carbonate rocks": "Cp",
            "acid rocks": "γ", "mafic rocks": "β", "peat": "Pt", "water": "", "metamorphic rocks": "M",
            "intermediate rocks": "δ", "gypsum evaporite": "Ev"}
    g["code"] = g["litho_irpi"].map(abbr).fillna("?")
    units, tab = build_units(g, "litho_irpi", "code", lambda d: pd.Series("", index=d.index), "litho_irpi")
    return Source("GEO-LiM", "Europe", units, tab, None, (90, 420))


def load_ethiopia(shp, simplify_m=40):
    g = _clean(_read(shp), "EPSG:32637", simplify_m)
    agef = lambda d: (d["ODD2_od"].fillna("").astype(str) + " " + d["UTVAR2_od"].fillna("").astype(str) + " "
                      + d["ERA2_od"].fillna("").astype(str) + " " + d["ERA1_do"].fillna("").astype(str))
    lithf = lambda d: d["HOR_TYP"].fillna("").astype(str) + " " + d["HOR_KARTO"].fillna("").astype(str)
    g["code"] = g["legend_loo"].fillna(g["IDX"]).astype(str)
    units, tab = build_units(g, "N_LEG_ID", "code", agef, lithf)
    return Source("UNESCO-Ethiopia", "Africa", units, tab, None, (90, 420))


GA_LAYERS = ["Pre_Neoproterozoic", "Neoproterozoic", "Paleozoic", "Mesozoic", "Cenozoic"]  # du plus vieux au plus jeune


def load_australia(folder, simplify_m=60):
    """Couches par ère EMPILÉES : on dessine la plus jeune par-dessus (ordre de peinture)."""
    folder = Path(folder)
    parts, flts = [], []
    for i, era in enumerate(GA_LAYERS):
        p = folder / f"{era}GeologicalUnitPolygons1M.shp"
        if p.exists():
            g = _clean(_read(p, columns=["MAPSYMBOL", "PLOTSYMBOL", "GEOLHIST", "LITHOLOGY"]), "EPSG:3577", simplify_m)
            g["order"] = i
            parts.append(g)
        q = folder / f"{era}ShearDisplacementLines1M.shp"
        if q.exists():
            f = _read(q, columns=["EXPOSURE", "MVTTYPE"]).to_crs("EPSG:3577")
            f["geometry"] = f.geometry.simplify(simplify_m)
            flts.append(f)
    g = pd.concat(parts, ignore_index=True)
    units, tab = build_units(g, "MAPSYMBOL", "MAPSYMBOL", "GEOLHIST", "LITHOLOGY", order=g["order"].values)
    f = pd.concat(flts, ignore_index=True)
    f = f[f.geometry.notnull() & ~f.geometry.is_empty].reset_index(drop=True)
    mv = f["MVTTYPE"].fillna("").str.lower()
    f["kind"] = np.where(mv.str.contains("thrust|reverse"), "thrust", np.where(mv.str.contains("normal"), "normal", "generic"))
    f["concealed"] = f["EXPOSURE"].fillna("").str.lower().str.contains("conceal|inferred|buried")
    return Source("GA-Australia", "Oceania", units, tab, gpd.GeoDataFrame(f[["kind", "concealed"]], geometry=f.geometry, crs=f.crs),
                  (100, 450))


# Codes de lignes des cartes d'état USGS OF 2005-1305 (métadonnées CAmetadata.txt)
ARC_FAULT = {4, 5, 10, 11, 30, 31, 32, 52, 53, 57, 58, 87, 88, 89, 90, 91, 92, 100, 109, 113, 114, 115, 116}
ARC_THRUST, ARC_NORMAL = {10, 11, 53}, {4, 5, 52}
ARC_CONCEALED = {32, 52, 53, 57, 58, 90, 92, 100, 115, 116}


def load_usgs_state(poly_shp, arc_shp, name, simplify_m=30):
    g = _clean(_read(poly_shp), "EPSG:5070", simplify_m)
    g = g[~g["ORIG_LABEL"].astype(str).str.lower().isin(["water", "h2o"])].reset_index(drop=True)
    units, tab = build_units(g, "ORIG_LABEL", "ORIG_LABEL", "UNIT_AGE",
                             lambda d: d["ROCKTYPE1"].fillna("").astype(str) + " " + d["ROCKTYPE2"].fillna("").astype(str))
    a = _read(arc_shp, columns=["ARC_CODE"]).to_crs("EPSG:5070")
    a = a[a["ARC_CODE"].isin(ARC_FAULT)].reset_index(drop=True)
    a["kind"] = np.where(a.ARC_CODE.isin(ARC_THRUST), "thrust", np.where(a.ARC_CODE.isin(ARC_NORMAL), "normal", "generic"))
    a["concealed"] = a.ARC_CODE.isin(ARC_CONCEALED)
    return Source(name, "North America", units, tab, a[["kind", "concealed", "geometry"]], (80, 380), min_land=0.8)


# =====================================================================================
#  RASTERISATION D'UNE FENÊTRE
# =====================================================================================
def raster_units(src, x0, y1, px, n):
    """Carte des uid (n x n) d'une fenêtre ; ordre de peinture respecté (jeune par-dessus)."""
    tr = from_origin(x0, y1, px, px)
    win = box(x0, y1 - n * px, x0 + n * px, y1)
    idx = src.units.sindex.query(win, predicate="intersects")
    if len(idx) == 0:
        return np.zeros((n, n), np.int32), win
    sel = src.units.iloc[idx]
    if sel["order"].nunique() > 1:
        sel = sel.sort_values("order", kind="stable")
    U = rasterize(zip(sel.geometry.values, sel.uid.values), out_shape=(n, n), transform=tr, fill=0, dtype="int32")
    return U, win


def fault_lines_px(src, win, x0, y1, px, rng=None):
    """Failles de la fenêtre en coordonnées pixel (à l'échelle px)."""
    out = []
    if src.faults is None or not len(src.faults):
        return out
    f = src.faults.iloc[src.faults.sindex.query(win.buffer(px * 4), predicate="intersects")]
    for geom, kind, conc in zip(f.geometry.values, f["kind"].values, f["concealed"].values):
        geom = geom.intersection(win.buffer(px * 4))
        for line in getattr(geom, "geoms", [geom]):
            if line.geom_type != "LineString" or line.length < px:
                continue
            xy = np.asarray(line.coords)[:, :2]
            out.append(((xy[:, 0] - x0) / px, (y1 - xy[:, 1]) / px, kind, bool(conc)))
    return out


MMU_AREA, MMU_RAD = 40, 1.5   # unité minimale de cartographie (px de la tuile finale) : aire et demi-épaisseur


def generalize(U, s=1, min_area=MMU_AREA, min_rad=MMU_RAD, passes=3):
    """Généralisation cartographique : tout polygone (composante connexe d'une même valeur) trop petit
    ou trop fin pour être visible sous les traits de contour est absorbé par ses voisins (pixel le plus proche).
    s = pixels internes par pixel final (rendu sur-échantillonné)."""
    U = np.asarray(U)
    for _ in range(passes):
        lab = cc_label(U, background=-1, connectivity=1)
        n = int(lab.max())
        if n <= 1:
            break
        area = np.bincount(lab.ravel(), minlength=n + 1)
        b = np.zeros(lab.shape, bool)
        b[:, 1:] |= lab[:, 1:] != lab[:, :-1]; b[:, :-1] |= lab[:, 1:] != lab[:, :-1]
        b[1:, :] |= lab[1:, :] != lab[:-1, :]; b[:-1, :] |= lab[1:, :] != lab[:-1, :]
        dist = ndi.distance_transform_edt(~b)
        rad = np.zeros(n + 1); rad[1:] = ndi.maximum(dist, lab, np.arange(1, n + 1))
        small = (area < min_area * s * s) | (rad < min_rad * s)
        small[0] = False
        rm = small[lab]
        if not rm.any() or rm.all():
            break
        _, (iy, ix) = ndi.distance_transform_edt(rm, return_indices=True)
        U = U[iy, ix]
    return U


def instances(U):
    """Composantes connexes d'une même unité -> instances (0 = vide)."""
    inst = cc_label(U, background=0, connectivity=1).astype(np.int32)
    return inst


def faults_gt(lines, scale=1.0, n=TILE):
    img = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(img)
    for xs, ys, _, _ in lines:
        pts = list(zip(xs * scale, ys * scale))
        if len(pts) > 1:
            d.line(pts, fill=1, width=1)
    return np.asarray(img, np.uint8)


# =====================================================================================
#  RENDU "CARTE GÉOLOGIQUE IMPRIMÉE"
# =====================================================================================
SEA_WORDS = ["PACIFIC OCEAN", "ATLANTIC OCEAN", "SEA", "GULF", "BAY", "BASIN", "LAKE", "STRAIT", "OCEAN",
             "INDIAN OCEAN", "CORAL SEA", "MER", "GOLFE", "LAC", "SOUND", "CHANNEL", "TROUGH", "ABYSSAL PLAIN"]
LAND_WORDS = ["MOUNTAINS", "PLATEAU", "RANGE", "BASIN", "SHIELD", "PLAIN", "DESERT", "VALLEY", "HIGHLANDS", "MASSIF"]
SYL = ["ka", "ri", "mo", "ton", "ber", "ville", "lin", "sa", "na", "port", "ford", "dor", "ve", "la", "mar", "zan",
       "gu", "el", "ash", "wa", "bur", "go", "ta", "ne", "kel", "an", "os", "ru", "mi", "hol"]


def _rname(rng, k=None):
    k = k or rng.integers(2, 4)
    return "".join(rng.choice(SYL) for _ in range(k)).capitalize()


def edition(rng):
    """Style d'une 'édition' de carte (tiré à chaque tuile)."""
    e = {}
    e["palette"] = "legend" if rng.random() < 0.8 else "random"
    e["light"] = rng.uniform(0.0, 0.4)
    e["sat"] = rng.uniform(0.7, 1.1)
    e["hue"] = rng.uniform(-0.03, 0.03)
    e["jit"] = rng.uniform(0.01, 0.06)
    e["sea"] = [(228, 229, 238), (214, 233, 245), (255, 255, 255), (236, 240, 232), (200, 222, 240)][rng.integers(5)]
    e["contacts"] = rng.random() > 0.06
    e["c_w"] = int(rng.choice([1, 2, 2, 2, 3]))                     # en px internes (x2)
    e["c_col"] = tuple(int(v) for v in rng.integers(20, 80, 3))
    e["c_dash"] = rng.uniform(0, 0.15)
    e["coast"] = rng.random() < 0.85
    e["coast_col"] = (int(rng.integers(40, 110)), int(rng.integers(140, 190)), int(rng.integers(200, 240)))
    e["f_w"] = int(rng.choice([3, 4, 5, 6]))
    e["f_col"] = (0, 0, 0) if rng.random() < 0.75 else ((int(rng.integers(150, 220)), 0, 0) if rng.random() < .7 else (60, 60, 60))
    e["pattern_p"] = rng.uniform(0.05, 0.35)
    e["label_p"] = rng.uniform(0.3, 0.95) if rng.random() < 0.9 else 0.0
    e["label_font"] = rng.choice(["sans", "sans", "bold", "serif", "ital"])
    e["label_size"] = rng.uniform(11, 22)
    e["label_col"] = tuple(int(v) for v in rng.integers(0, 70, 3))
    e["scan"] = rng.random() < 0.35
    return e


def _palette(src, uids, e, rng, ed_seed):
    lut = {}
    if e["palette"] == "mono":     # piège : TOUTES les unités de la même couleur, seuls les contacts les séparent
        c = np.array(colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.15, 0.5), rng.uniform(0.75, 0.95)), np.float32) * 255
        return {u: c.copy() for u in uids}
    if e["palette"] == "random":   # couleurs bien séparées, attribuées au hasard (le réseau n'apprend pas "couleur = unité")
        n = max(len(uids), 1)
        style = rng.choice(["pastel", "vif", "moyen"])
        s_lo, s_hi, v_lo, v_hi = {"pastel": (0.25, 0.45, 0.85, 0.97), "vif": (0.5, 0.75, 0.8, 0.95),
                                  "moyen": (0.35, 0.65, 0.65, 0.9)}[style]
        off = rng.random()
        hues = [(off + i / n) % 1.0 for i in range(n)]
        rng.shuffle(hues)
        for i, (u, h) in enumerate(zip(uids, hues)):
            v = v_hi if i % 2 == 0 else v_lo
            lut[u] = np.array(colorsys.hsv_to_rgb(h, rng.uniform(s_lo, s_hi), v), np.float32) * 255
        return lut
    for u in uids:
        r = np.random.default_rng(_seed(ed_seed, src.name, u))
        h, s, v = colorsys.rgb_to_hsv(*(src.lut_rgb[u] / 255))
        h = (h + e["hue"] + r.normal(0, e["jit"] * 0.5)) % 1
        s = np.clip(s * e["sat"] * r.uniform(1 - e["jit"] * 3, 1 + e["jit"] * 2), 0, 1)
        v = np.clip(v * r.uniform(1 - e["jit"] * 2, 1 + e["jit"]), 0, 1)
        c = np.array(colorsys.hsv_to_rgb(h, s, v), np.float32) * 255
        lut[u] = _mix(c, (255, 255, 255), np.clip(e["light"] + r.normal(0, 0.05), 0, 0.7))
    return lut


def _glyph_tile(kind, s, rng):
    """Motif élémentaire (s x s) répété par np.tile -> masque booléen R x R."""
    im = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(im)
    c, k = s / 2, s * rng.uniform(0.18, 0.3)
    if kind == "dots":
        r = max(1.2, s * rng.uniform(0.07, 0.14))
        d.ellipse([c - r, c - r, c + r, c + r], fill=1)
    elif kind == "v":
        d.line([(c - k, c - k), (c, c + k), (c + k, c - k)], fill=1, width=2)
    elif kind == "tri":
        d.polygon([(c - k, c + k), (c + k, c + k), (c, c - k)], fill=1)
    elif kind == "cross":
        d.line([(c - k, c), (c + k, c)], fill=1, width=2)
        d.line([(c, c - k), (c, c + k)], fill=1, width=2)
    elif kind == "dash":
        d.line([(c - k, c), (c + k, c)], fill=1, width=2)
    m = np.asarray(im, bool)
    m2 = np.concatenate([m, np.roll(m, s // 2, axis=1)], 0)   # rangées décalées
    full = np.tile(m2, (R // (2 * s) + 2, R // s + 2))
    oy, ox = rng.integers(0, 2 * s), rng.integers(0, s)
    return full[oy:oy + R, ox:ox + R]


def _pattern_mask(kind, rng):
    yy, xx = np.mgrid[:R, :R]
    if kind == "lines":
        per, w = rng.integers(14, 30), rng.integers(2, 4)
        a = rng.choice([np.pi / 4, -np.pi / 4, 0, np.pi / 2, np.pi / 3])
        return ((xx * np.cos(a) + yy * np.sin(a)) % per) < w
    if kind == "grid":
        per = rng.integers(18, 34)
        return ((xx % per) < 2) | ((yy % per) < 2)
    return _glyph_tile(kind, int(rng.integers(14, 30)), rng)


def _resample(xs, ys, step):
    seg = np.hypot(np.diff(xs), np.diff(ys))
    L = np.concatenate([[0], np.cumsum(seg)])
    if L[-1] < step:
        return None
    t = np.arange(0, L[-1], step)
    return np.interp(t, L, xs), np.interp(t, L, ys)


def _draw_fault(d, xs, ys, kind, concealed, w, col, rng, side):
    pts = list(zip(xs, ys))
    if concealed:
        rs = _resample(xs, ys, 3)
        if rs is None:
            return
        X, Y = rs
        on, off = (8, 6) if rng.random() < 0.6 else (2, 4)
        i = 0
        while i < len(X) - 1:
            j = min(i + on, len(X) - 1)
            d.line(list(zip(X[i:j + 1], Y[i:j + 1])), fill=col, width=max(2, w - 1))
            i = j + off
    else:
        d.line(pts, fill=col, width=w, joint="curve")
    if kind in ("thrust", "normal"):
        rs = _resample(xs, ys, 2)
        if rs is None:
            return
        X, Y = rs
        gap = int(rng.integers(14, 24)) if kind == "thrust" else int(rng.integers(30, 50))
        for i in range(gap // 2, len(X) - 3, gap):
            tx, ty = X[i + 2] - X[i], Y[i + 2] - Y[i]
            n = math.hypot(tx, ty) or 1
            tx, ty = tx / n, ty / n
            nx, ny = -ty * side, tx * side
            h = w * 1.9
            if kind == "thrust":
                d.polygon([(X[i] - tx * h * .7, Y[i] - ty * h * .7), (X[i] + tx * h * .7, Y[i] + ty * h * .7),
                           (X[i] + nx * h * 1.3, Y[i] + ny * h * 1.3)], fill=col)
            else:
                ex, ey = X[i] + nx * h * 1.6, Y[i] + ny * h * 1.6
                d.line([(X[i], Y[i]), (ex, ey)], fill=col, width=max(2, w // 2))
                r = w * 0.8
                d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=col)


def _random_curve(rng, wiggle=0.12, step=6):
    side = rng.integers(4)
    p = rng.uniform(0, R)
    x, y, a = [(p, 0, np.pi / 2), (R, p, np.pi), (p, R, -np.pi / 2), (0, p, 0)][side]
    a += rng.uniform(-0.7, 0.7)
    xs, ys, da = [x], [y], 0.0
    for _ in range(4 * R // step):
        da = 0.85 * da + rng.normal(0, wiggle)
        a += da * 0.3
        x, y = x + step * np.cos(a), y + step * np.sin(a)
        xs.append(x); ys.append(y)
        if not (-20 < x < R + 20 and -20 < y < R + 20):
            break
    return np.array(xs), np.array(ys)


def _dashed(d, xs, ys, col, w, on, off):
    rs = _resample(xs, ys, 2)
    if rs is None:
        return
    X, Y = rs
    i = 0
    while i < len(X) - 1:
        j = min(i + on, len(X) - 1)
        d.line(list(zip(X[i:j + 1], Y[i:j + 1])), fill=col, width=w)
        i = j + off


def _spaced(t):
    return " ".join(t) if len(t) < 12 else t


# Classes d'encre (mode layers) : 0 aucune, 1 contact, 2 côte, 3 faille, 4 texte / étiquette / trait de rappel,
# 5 autre trait (isobathe, graticule, rivière, frontière). Le fond topographique (topo_aug) ajoute la classe 6.
INK_CLASSES = ["none", "contact", "coast", "fault", "text", "other_line", "topo"]


def render_map(src, U, lines, rng, px_m, overrides=None, layers=False):
    """U : uid (R x R) ; lines : failles en px internes. Retourne l'image RGB uint8 (TILE x TILE).
    overrides : force des paramètres d'édition (ex. palette='mono', pattern_p=1.0 pour les jeux 'pièges').
    layers=True : renvoie en plus ed["_plate"] (la même carte sans aucune encre, mêmes impression et numérisation) et
    ed["_ink"] (classe d'encre par pixel, voir INK_CLASSES). Les tirages aléatoires sont identiques à layers=False."""
    e = edition(rng)
    e.update(overrides or {})
    ed_seed = int(rng.integers(1 << 30))
    uids = [int(u) for u in np.unique(U) if u > 0]
    lut = _palette(src, uids, e, rng, ed_seed)
    LUT = np.zeros((max(uids + [0]) + 1, 3), np.float32)
    LUT[0] = e["sea"]
    for u, c in lut.items():
        LUT[u] = c
    img = LUT[U]
    land = U > 0

    # --- motifs (pointillés, hachures, V volcaniques, croix plutoniques, triangles)
    masks = {}
    for u in uids:
        pref = src.pref.get(u)
        r = np.random.default_rng(_seed(ed_seed, "pat", u))
        p = e["pattern_p"] * (2.2 if pref else 1)
        if p < 1 and r.random() > p:
            continue
        kind = pref if (pref and r.random() < 0.7) else r.choice(["dots", "lines", "lines", "v", "tri", "dash", "grid", "cross"])
        if kind not in masks:
            masks[kind] = _pattern_mask(kind, rng)
        m = (U == u) & masks[kind]
        if kind == "tri" and r.random() < 0.5:
            col = np.array([r.integers(180, 240), r.integers(0, 60), r.integers(40, 100)], np.float32)
        elif r.random() < 0.3:
            col = np.array([r.integers(120, 180), r.integers(60, 110), r.integers(20, 60)], np.float32)
        else:
            col = LUT[u] * r.uniform(0.45, 0.75)
        img[m] = col

    plate = img.copy() if layers else None
    ink = np.zeros(U.shape, np.uint8) if layers else None

    # --- contacts géologiques
    if e["contacts"]:
        b = np.zeros_like(land)
        b[:, 1:] |= U[:, 1:] != U[:, :-1]
        b[1:, :] |= U[1:, :] != U[:-1, :]
        b &= land | np.roll(land, 1, 0) | np.roll(land, 1, 1)
        coast = np.zeros_like(land)
        coast[:, 1:] |= land[:, 1:] != land[:, :-1]
        coast[1:, :] |= land[1:, :] != land[:-1, :]
        b &= ~coast
        if e["c_w"] > 1:
            b = ndi.binary_dilation(b, iterations=e["c_w"] - 1)
        if e["c_dash"] > 0.02:
            dashed_units = [u for u in uids if np.random.default_rng(_seed(ed_seed, "d", u)).random() < e["c_dash"]]
            if dashed_units:
                yy, xx = np.mgrid[:R, :R]
                dz = ndi.binary_dilation(np.isin(U, dashed_units), iterations=2)
                b &= ~(dz & (((xx + yy) // 10) % 2 == 0))
        img[b] = e["c_col"]
        if layers:
            ink[b] = 1
        if e["coast"]:
            cw = int(rng.integers(2, 5))
            cm = ndi.binary_dilation(coast, iterations=cw - 1) if cw > 1 else coast
            img[cm] = e["coast_col"]
            if layers:
                ink[cm] = 2

    sea_frac = 1 - land.mean()
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    last = [np.asarray(im).copy()] if layers else None

    def mark(cls):          # mode layers : les pixels modifiés depuis le dernier appel reçoivent la classe cls
        if layers:
            a = np.asarray(im)
            ink[(a != last[0]).any(-1)] = cls
            last[0] = a.copy()

    texts = []   # tous les textes imprimés : {"k": "u" (code d'unité) / "o" (autre), "b": boîte, "t": point désigné}

    def note(kind, xy, t, f, target=None, **kw):
        b = d.textbbox(xy, t, font=f, **kw)
        rec = {"k": kind, "b": [round(v / SS, 1) for v in b], "s": t}
        if target is not None:
            rec["t"] = [round(target[0] / SS, 1), round(target[1] / SS, 1)]
        texts.append(rec)

    # --- mer : isobathes + grands noms bleus
    if sea_frac > 0.1 and rng.random() < 0.6:
        noise = ndi.gaussian_filter(rng.normal(size=(R // 8, R // 8)), rng.uniform(3, 7))
        noise = ndi.zoom(noise, 8, order=1)
        col = (int(rng.integers(90, 160)), int(rng.integers(170, 210)), int(rng.integers(215, 245)))
        for lev in np.quantile(noise, rng.uniform(0.2, 0.8, rng.integers(1, 4))):
            for c in find_contours(noise, lev):
                if len(c) < 30:
                    continue
                keep = ~land[np.clip(c[:, 0].astype(int), 0, R - 1), np.clip(c[:, 1].astype(int), 0, R - 1)]
                c = c[keep]
                if len(c) > 30:
                    _dashed(d, c[:, 1], c[:, 0], col, 2, 10, 6) if rng.random() < .5 else d.line(list(zip(c[:, 1], c[:, 0])), fill=col, width=2)
    if sea_frac > 0.25 and rng.random() < 0.6:
        ys, xs = np.nonzero(~land[::16, ::16])
        if len(ys):
            k = rng.integers(len(ys))
            t = rng.choice(SEA_WORDS)
            t = _spaced(t) if rng.random() < .7 else t
            f = font(rng.choice(["bital", "serifi", "ital"]), rng.uniform(30, 70))
            xy = (xs[k] * 16 - rng.uniform(0, 300), ys[k] * 16)
            d.text(xy, t, font=f, fill=(int(rng.integers(20, 80)), int(rng.integers(80, 140)), int(rng.integers(170, 220))))
            note("o", xy, t, f)

    mark(5)
    # --- graticule (méridiens / parallèles) + degrés
    for _ in range(rng.integers(0, 3)):
        col = tuple(int(v) for v in (rng.integers(0, 90, 3) if rng.random() < .6 else (rng.integers(150, 210), 120, rng.integers(150, 210))))
        a, b2 = rng.uniform(0, R, 2)
        if rng.random() < 0.5:
            d.line([(a, 0), (b2, R)], fill=col, width=int(rng.integers(1, 3)))
            lab_xy = (a + 6, rng.uniform(20, R - 40))
        else:
            d.line([(0, a), (R, b2)], fill=col, width=int(rng.integers(1, 3)))
            lab_xy = (rng.uniform(20, R - 80), a + 6)
        if rng.random() < 0.5:
            t, f = f"{rng.integers(0, 180)}°", font("bold", rng.uniform(22, 40))
            d.text(lab_xy, t, font=f, fill=col)
            note("o", lab_xy, t, f)

    mark(5)
    # --- rivières
    if rng.random() < 0.55:
        col = (int(rng.integers(60, 130)), int(rng.integers(150, 200)), int(rng.integers(210, 245)))
        for _ in range(rng.integers(1, 5)):
            xs, ys = _random_curve(rng, wiggle=0.2, step=5)
            d.line(list(zip(xs, ys)), fill=col, width=int(rng.integers(2, 4)), joint="curve")

    mark(5)
    # --- frontières politiques (tirets colorés)
    if rng.random() < 0.35:
        col = [(200, 120, 200), (170, 110, 60), (150, 90, 170), (230, 150, 60), (120, 120, 120)][rng.integers(5)]
        xs, ys = _random_curve(rng, wiggle=0.08, step=8)
        _dashed(d, xs, ys, col, int(rng.integers(4, 9)), int(rng.integers(8, 20)), int(rng.integers(4, 10)))

    mark(5)
    # --- failles
    for xs, ys, kind, conc in lines:
        r = np.random.default_rng(_seed(ed_seed, "f", round(float(xs[0]), 1), round(float(ys[0]), 1)))
        if kind == "generic" and r.random() < 0.15:
            kind = r.choice(["thrust", "normal"])
        _draw_fault(d, xs, ys, kind, conc, e["f_w"], e["f_col"], r, 1 if r.random() < .5 else -1)

    mark(3)
    # --- étiquettes d'unités (au point le plus intérieur de chaque instance, + traits de rappel)
    if e["label_p"] > 0 and uids:
        inst = cc_label(U, background=0, connectivity=1)
        n = inst.max()
        if n:
            dist = ndi.distance_transform_edt(inst > 0)
            idx = np.arange(1, n + 1)
            pos = ndi.maximum_position(dist, inst, idx)
            val = ndi.maximum(dist, inst, idx)
            uid_of = ndi.maximum(U, inst, idx)
            fs = int(e["label_size"] * SS)
            f = font(e["label_font"], fs)
            placed = []
            for (py, px), v, u in zip(pos, val, uid_of):
                if rng.random() > e["label_p"]:
                    continue
                t = str(src.codes.get(int(u), ""))[:8]
                if not t or t == "nan":
                    continue
                if e["label_font"] == "bold" and rng.random() < .3:
                    t = t.upper()
                tw = d.textlength(t, font=f)
                r_ = rng.random()
                if v > fs * 0.8:                                   # l'étiquette tient dans le polygone
                    x, y = px - tw / 2, py - fs / 2
                elif v > 2 and r_ < 0.45:                          # trop petit : étiquette à côté + trait de rappel
                    ang = rng.uniform(0, 2 * np.pi)
                    L = rng.uniform(30, 110)
                    cx_, cy_ = px + L * np.cos(ang), py + L * np.sin(ang)
                    x, y = cx_ - tw / 2, cy_ - fs / 2
                    ax_ = min(max(px, x), x + tw)                  # point d'attache : bord de la boîte le plus proche
                    ay_ = min(max(py, y), y + fs)
                    d.line([(px, py), (ax_, ay_)], fill=e["label_col"], width=2)
                elif v > 2 and r_ < 0.75:                          # trop petit : étiquette qui déborde du polygone
                    x, y = px - tw / 2 + rng.uniform(-tw / 3, tw / 3), py - fs / 2 + rng.uniform(-fs / 3, fs / 3)
                else:
                    continue
                bb = (x - 2, y - 2, x + tw + 2, y + fs + 2)
                if any(not (bb[2] < q[0] or bb[0] > q[2] or bb[3] < q[1] or bb[1] > q[3]) for q in placed):
                    continue
                placed.append(bb)
                if rng.random() < 0.15:   # halo blanc
                    d.text((x, y), t, font=f, fill=e["label_col"], stroke_width=3, stroke_fill=(255, 255, 255))
                else:
                    d.text((x, y), t, font=f, fill=e["label_col"])
                note("u", (x, y), t, f, target=(px, py))

    # --- villes et grands noms de régions (distracteurs)
    for _ in range(rng.integers(0, 4)):
        x, y = rng.uniform(20, R - 200), rng.uniform(20, R - 40)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(0, 0, 0), outline=(255, 255, 255))
        t, f = _rname(rng), font(rng.choice(["sans", "ital"]), rng.uniform(18, 26))
        d.text((x + 10, y - 14), t, font=f, fill=(20, 20, 20))
        note("o", (x + 10, y - 14), t, f)
    if rng.random() < 0.15:
        t, f = _spaced(rng.choice(LAND_WORDS)), font("bold", rng.uniform(28, 50))
        xy = (rng.uniform(0, R / 2), rng.uniform(0, R - 60))
        d.text(xy, t, font=f, fill=(int(rng.integers(80, 140)), int(rng.integers(50, 90)), int(rng.integers(20, 60))))
        note("o", xy, t, f)

    mark(4)
    # --- réduction x2 (anti-crénelage d'impression)
    flt_ = Image.LANCZOS if rng.random() < .5 else Image.BOX
    im = im.resize((TILE, TILE), flt_)
    if layers:
        pim = Image.fromarray(np.clip(plate, 0, 255).astype(np.uint8)).resize((TILE, TILE), flt_)
        k = R // TILE                     # classe d'encre : la plus « forte » du bloc k x k (priorité au numéro le plus haut)
        ink = ink.reshape(TILE, k, TILE, k).max(axis=(1, 3))

    # --- impression / numérisation (mêmes tirages, appliqués à la carte et, en mode layers, à la plaque)
    ims = [im, pim] if layers else [im]
    if e["scan"]:
        paper = np.array([rng.uniform(238, 255), rng.uniform(230, 252), rng.uniform(205, 240)], np.float32) / 255
        gy, gx = np.mgrid[:TILE, :TILE] / TILE
        grad = (1 - rng.uniform(0, 0.08) * (rng.uniform(-1, 1) * gx + rng.uniform(-1, 1) * gy))[..., None]
        sig, gam = rng.uniform(0.2, 0.9), None
        out = []
        for j, x in enumerate(ims):
            a = np.asarray(x, np.float32) * paper * grad
            a = ndi.gaussian_filter(a, (sig,) * 2 + (0,))
            if j == 0:
                gam = rng.uniform(0.85, 1.15)
                noise = rng.normal(0, rng.uniform(0.5, 4), a.shape)
                shift = (rng.integers(3), rng.choice([-1, 1]), rng.integers(2)) if rng.random() < 0.2 else None
            a = 255 * (np.clip(a, 0, 255) / 255) ** gam + noise
            if shift is not None:
                a[..., shift[0]] = np.roll(a[..., shift[0]], shift[1], axis=shift[2])
            out.append(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
        ims = out
        q = int(rng.integers(50, 92))
    else:
        q = int(rng.integers(80, 98)) if rng.random() < .5 else None
    if q:
        for j, x in enumerate(ims):
            buf = io.BytesIO()
            x.save(buf, "JPEG", quality=q)
            ims[j] = Image.open(buf).convert("RGB")
    im = ims[0]
    ed = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in e.items()}
    ed["_texts"] = texts
    if layers:
        ed["_plate"], ed["_ink"] = np.asarray(ims[1], np.uint8), ink
    return np.asarray(im, np.uint8), ed


# =====================================================================================
#  ÉCHANTILLONNAGE SPATIAL (split train / val / test par blocs géographiques)
# =====================================================================================
class BlockSplit:
    def __init__(self, src, block_m, seed, val=0.1, test=0.1, min_land=0.3):
        x0, y0, x1, y1 = src.units.total_bounds
        self.x0, self.y0, self.b = x0, y0, block_m
        nx, ny = int(np.ceil((x1 - x0) / block_m)), int(np.ceil((y1 - y0) / block_m))
        rng = np.random.default_rng(seed)
        # part de terre de chaque bloc, sur un masque grossier (5 km)
        cell = 5000.0
        k = int(round(block_m / cell))
        mask = rasterize(((g, 1) for g in src.units.geometry.values), out_shape=(ny * k, nx * k),
                         transform=from_origin(x0, y0 + ny * block_m, cell, cell), fill=0, dtype="uint8")
        frac = mask.reshape(ny, k, nx, k).mean(axis=(1, 3))[::-1]      # ligne 0 = sud
        land = [(i, j) for i in range(nx) for j in range(ny) if frac[j, i] >= min_land]
        order = rng.permutation(len(land))
        n_val, n_test = max(1, round(val * len(land))), max(1, round(test * len(land)))
        self.split = {}
        for k, o in enumerate(order):
            self.split[land[o]] = "validation" if k < n_val else ("test" if k < n_val + n_test else "train")
        self.blocks = {s: [b for b, v in self.split.items() if v == s] for s in ("train", "validation", "test")}

    def window_split(self, x0, y0, x1, y1):
        i0, i1 = int((x0 - self.x0) // self.b), int((x1 - self.x0) // self.b)
        j0, j1 = int((y0 - self.y0) // self.b), int((y1 - self.y0) // self.b)
        s = {self.split.get((i, j), "none") for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}
        s.discard("none")
        return s.pop() if len(s) == 1 else None


def synth_tile(src, bs, split, rng, max_tries=60, overrides=None, layers=False):
    """Tire une fenêtre entièrement contenue dans des blocs du split demandé, et la rend."""
    lo, hi = src.px_range
    for _ in range(max_tries):
        hi_eff = min(hi, 0.9 * bs.b / TILE)          # la fenêtre doit pouvoir tenir dans un bloc
        px = float(np.exp(rng.uniform(np.log(min(lo, hi_eff)), np.log(hi_eff))))
        ext = TILE * px
        bi, bj = bs.blocks[split][rng.integers(len(bs.blocks[split]))]
        if rng.random() < 0.5:                        # entièrement dans le bloc tiré
            x0 = bs.x0 + bi * bs.b + rng.random() * (bs.b - ext)
            y1 = bs.y0 + bj * bs.b + ext + rng.random() * (bs.b - ext)
        else:                                         # libre (peut déborder sur des blocs du même split)
            cx = bs.x0 + (bi + rng.random()) * bs.b
            cy = bs.y0 + (bj + rng.random()) * bs.b
            x0, y1 = cx - rng.uniform(0.1, 0.9) * ext, cy + rng.uniform(0.1, 0.9) * ext
        if bs.window_split(x0, y1 - ext, x0 + ext, y1) != split:
            continue
        U, win = raster_units(src, x0, y1, px / SS, R)
        U = generalize(U, SS)
        land = (U > 0).mean()
        if land < src.min_land:
            continue
        nu = len(np.unique(U[U > 0]))
        if nu < 2 or (nu < 4 and rng.random() < 0.6):
            continue
        lines = fault_lines_px(src, win, x0, y1, px / SS)
        rgb, ed = render_map(src, U, lines, rng, px, overrides, layers)
        Uf = generalize(U[SS // 2::SS, SS // 2::SS], 1)   # le sous-échantillonnage peut couper des isthmes fins
        inst = instances(Uf)
        flt = faults_gt(lines, 1 / SS)
        texts = ed.pop("_texts", [])
        meta = {"bounds": [x0, y1 - ext, x0 + ext, y1], "px_size_m": px, "edition": ed, "texts": texts}
        return rgb, inst, Uf, flt, meta
    return None


def inst_units(src, inst, Uf):
    """Pour chaque instance : son unité (code, âge, lithologie)."""
    n = int(inst.max())
    if n == 0:
        return {}
    uid = ndi.maximum(Uf, inst, np.arange(1, n + 1)).astype(int)
    return {str(i + 1): src.unit_meta(u) for i, u in enumerate(uid)}


# =====================================================================================
#  ENCODAGE (format publiable : PNG dans Parquet)
# =====================================================================================
def png_bytes(arr, mode=None):
    buf = io.BytesIO()
    if arr.dtype == np.uint16 or mode == "I;16":
        Image.fromarray(arr.astype(np.uint16)).save(buf, "PNG", optimize=False)
    else:
        Image.fromarray(arr).save(buf, "PNG", optimize=False)
    return buf.getvalue()


def make_record(tid, source, continent, kind, split, crs, rgb, inst, flt, meta, units):
    if inst.max() > 65535:
        raise ValueError("trop d'instances")
    return {
        "id": tid, "source": source, "continent": continent, "kind": kind, "split": split,
        "image": {"bytes": png_bytes(rgb), "path": f"{tid}.png"},
        "instances": {"bytes": png_bytes(inst.astype(np.uint16)), "path": f"{tid}_inst.png"},
        "faults": {"bytes": png_bytes((flt > 0).astype(np.uint8) * 255), "path": f"{tid}_faults.png"},
        "n_instances": int(inst.max()),
        "land_fraction": float((inst > 0).mean()),
        "fault_pixels": int((flt > 0).sum()),
        "px_size_m": float(meta["px_size_m"]),
        "crs": str(crs),
        "bounds": [float(v) for v in meta["bounds"]],
        "units": json.dumps(units, ensure_ascii=False),
        "render": json.dumps(meta.get("edition", {}), ensure_ascii=False),
        "texts": json.dumps(meta.get("texts", []), ensure_ascii=False),
    }


# =====================================================================================
#  CONSTRUCTION DU DATASET UNIQUE (réel + synthétique, 4 continents) -> Parquet
# =====================================================================================
import os, time, multiprocessing as mp
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import Window

SPLITS = ("train", "validation", "test")
_IMG = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
_FIELDS = [("id", pa.string()), ("source", pa.string()), ("continent", pa.string()), ("kind", pa.string()),
           ("split", pa.string()), ("image", _IMG), ("instances", _IMG), ("faults", _IMG),
           ("n_instances", pa.int32()), ("land_fraction", pa.float32()), ("fault_pixels", pa.int32()),
           ("px_size_m", pa.float32()), ("crs", pa.string()), ("bounds", pa.list_(pa.float64())),
           ("units", pa.string()), ("render", pa.string()), ("texts", pa.string())]


def arrow_schema():
    """Schéma Parquet + métadonnées 'huggingface' (le viewer HF affiche alors les images)."""
    try:
        from datasets import Features, Value, Image as HFImage, Sequence
        f = Features({"id": Value("string"), "source": Value("string"), "continent": Value("string"),
                      "kind": Value("string"), "split": Value("string"), "image": HFImage(),
                      "instances": HFImage(), "faults": HFImage(), "n_instances": Value("int32"),
                      "land_fraction": Value("float32"), "fault_pixels": Value("int32"),
                      "px_size_m": Value("float32"), "crs": Value("string"), "bounds": Sequence(Value("float64")),
                      "units": Value("string"), "render": Value("string"), "texts": Value("string")})
        return f.arrow_schema
    except Exception as ex:
        print("datasets indisponible, schéma sans métadonnées HF :", ex)
        return pa.schema(_FIELDS)


class ShardWriter:
    def __init__(self, out_dir, shard_rows=500):
        self.dir = Path(out_dir) / "data"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.schema, self.n = arrow_schema(), shard_rows
        from collections import defaultdict
        self.buf, self.k, self.count = defaultdict(list), defaultdict(int), defaultdict(int)
        self.index = []

    def add(self, rec):
        s = rec["split"]
        self.buf[s].append(rec)
        self.count[s] += 1
        self.index.append({k: rec[k] for k in ("id", "source", "continent", "kind", "split", "n_instances",
                                                "land_fraction", "fault_pixels", "px_size_m")})
        if len(self.buf[s]) >= self.n:
            self.flush(s)

    def flush(self, s):
        if not self.buf[s]:
            return
        cols = {name: [r[name] for r in self.buf[s]] for name in self.schema.names}
        tbl = pa.Table.from_pydict(cols, schema=self.schema)
        pq.write_table(tbl, self.dir / f"{s}-{self.k[s]:05d}.parquet", compression="zstd")
        self.k[s] += 1
        self.buf[s] = []

    def close(self):
        for s in list(self.buf):
            self.flush(s)
        df = pd.DataFrame(self.index)
        df.to_csv(self.dir.parent / "index.csv", index=False)
        return df


def real_tiles(raster_path, src, bs, name, writer, easy_frac=0.2, min_cover=0.5, seed=0):
    """Tuiles réelles (scan + vecteurs alignés), grille sans recouvrement, split par blocs géographiques."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    with rasterio.open(raster_path) as s:
        cands = []
        for r in range(0, s.height - TILE + 1, TILE):
            for c in range(0, s.width - TILE + 1, TILE):
                win = Window(c, r, TILE, TILE)
                x0, y0, x1, y1 = rasterio.windows.bounds(win, s.transform)
                split = bs.window_split(x0, y0, x1, y1)
                if split is None:
                    continue
                rgb = np.transpose(s.read([1, 2, 3], window=win), (1, 2, 0))
                if rgb.mean() > 248 or rgb.mean() < 5:
                    continue
                tr = rasterio.windows.transform(win, s.transform)
                U, w = raster_units(src, tr.c, tr.f, tr.a, TILE)
                if (U > 0).mean() < min_cover:
                    continue
                nu = len(np.unique(U[U > 0]))
                lines = fault_lines_px(src, w, tr.c, tr.f, tr.a)
                nf = int(faults_gt(lines, 1.0).sum())
                cands.append((r, c, split, nu >= 3 or nf > 30))
        rich = [k for k in cands if k[3]]
        easy = [k for k in cands if not k[3]]
        n_easy = min(len(easy), int(round(easy_frac / (1 - easy_frac) * len(rich))))
        easy = [easy[i] for i in rng.choice(len(easy), n_easy, replace=False)] if n_easy else []
        n = {sp: 0 for sp in SPLITS}
        for r, c, split, _ in rich + easy:
            win = Window(c, r, TILE, TILE)
            tr = rasterio.windows.transform(win, s.transform)
            rgb = np.ascontiguousarray(np.transpose(s.read([1, 2, 3], window=win), (1, 2, 0)))
            U, w = raster_units(src, tr.c, tr.f, tr.a, TILE)
            U = generalize(U, 1)
            lines = fault_lines_px(src, w, tr.c, tr.f, tr.a)
            inst, flt = instances(U), faults_gt(lines, 1.0)
            meta = {"bounds": list(rasterio.windows.bounds(win, s.transform)), "px_size_m": abs(tr.a),
                    "edition": {"raster": Path(raster_path).name, "row": r, "col": c}}
            writer.add(make_record(f"{name}_r{r}_c{c}", src.name, src.continent, "real", split, src.crs,
                                   rgb, inst, flt, meta, inst_units(src, inst, U)))
            n[split] += 1
    print(f"  {name} : {sum(n.values())} tuiles réelles {n} en {time.time() - t0:.0f} s", flush=True)
    return n


_SOURCES, _SPLITTERS = {}, {}


def _synth_job(job):
    name, split, i, seed = job
    src, bs = _SOURCES[name], _SPLITTERS[name]
    rng = np.random.default_rng(seed)
    for _ in range(30):
        res = synth_tile(src, bs, split, rng)
        if res is not None:
            rgb, inst, Uf, flt, meta = res
            tid = f"{src.name}_synth_{split}_{i:05d}"
            return make_record(tid, src.name, src.continent, "synthetic", split, src.crs, rgb, inst, flt, meta,
                               inst_units(src, inst, Uf))
    return None


def build_synthetic(sources, splitters, counts, writer, val=0.1, test=0.1, seed=2026, workers=None):
    _SOURCES.update(sources)
    _SPLITTERS.update(splitters)
    jobs = []
    for name, n in counts.items():
        if name not in sources:
            continue
        per = {"validation": int(round(n * val)), "test": int(round(n * test))}
        per["train"] = n - per["validation"] - per["test"]
        for split, k in per.items():
            jobs += [(name, split, i, _seed(seed, name, split, i)) for i in range(k)]
    np.random.default_rng(seed).shuffle(jobs)
    workers = workers or os.cpu_count()
    t0, done, fail = time.time(), 0, 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for rec in pool.imap_unordered(_synth_job, jobs, chunksize=4):
            if rec is None:
                fail += 1
                continue
            writer.add(rec)
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                print(f"  synthétique : {done}/{len(jobs)}  ({el:.0f} s, reste ~{el / done * (len(jobs) - done):.0f} s)", flush=True)
    print(f"  synthétique terminé : {done} tuiles, {fail} échecs, {time.time() - t0:.0f} s", flush=True)


def decode(rec):
    """Ligne Parquet -> (rgb uint8 HxWx3, inst int32 HxW, faults uint8 HxW {0,1}). Utilisé à l'entraînement."""
    rgb = np.asarray(Image.open(io.BytesIO(rec["image"]["bytes"])).convert("RGB"))
    inst = np.asarray(Image.open(io.BytesIO(rec["instances"]["bytes"]))).astype(np.int32)
    flt = (np.asarray(Image.open(io.BytesIO(rec["faults"]["bytes"]))) > 0).astype(np.uint8)
    return rgb, inst, flt
