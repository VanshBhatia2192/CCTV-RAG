import re
import torch
from src.common import clip
from src.incident import embed as iembed
from src.index import vector_db
from src.captions import text_index as T

_ft = {}
COLORS = {"black", "white", "gray", "red", "blue", "green", "yellow", "brown", "orange", "pink"}
UPPER = {"jacket", "shirt", "top", "hoodie", "coat", "tshirt", "t-shirt", "dress", "sweater"}
LOWER = {"jeans", "pants", "trousers", "shorts", "skirt"}
VTYPE = {"car": "car", "sedan": "car", "suv": "suv", "van": "van", "truck": "truck", "pickup": "truck",
         "bus": "bus", "motorcycle": "motorcycle", "motorbike": "motorcycle", "bike": "motorcycle"}
STILL = {"stopped", "parked", "stationary", "standing", "idle"}

def attr_bonus(kind, text, p, unit=0.01):
    t = text.lower().replace("grey", "gray")
    w = re.findall(r"[a-z\-]+", t); ws = set(w); b = 0.0
    for i, x in enumerate(w):
        if x in COLORS:
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if kind == "vehicle":
                b += p.get("color") == x
            elif nxt in LOWER:
                b += p.get("lower_color") == x
            elif nxt in UPPER:
                b += p.get("upper_color") == x
            else:
                b += x in (p.get("upper_color"), p.get("lower_color"))
        if kind == "vehicle" and x in VTYPE:
            b += p.get("type") == VTYPE[x]
    m = re.search(r"turn(?:ing|s)? (left|right)", t)
    if kind == "vehicle" and m:
        b += 1.5 * (p.get("turn") == m.group(1))
    if ws & STILL:
        b += p.get("motion") in ("stopped", "standing")
    elif ws & {"moving", "driving", "drives"}:
        b += p.get("motion") == "moving"
    elif "running" in ws:
        b += p.get("motion") == "running"
    elif "walking" in ws:
        b += p.get("motion") == "walking"
    if kind == "person":
        if "backpack" in ws:
            b += p.get("bag") == "backpack"
        elif ws & {"bag", "handbag", "purse"}:
            b += p.get("bag") in ("handbag", "shopping bag", "backpack")
        if ws & {"hat", "cap"}:
            b += p.get("hat") == "hat"
    return b * unit

def _visual(kind, text, k, where, ids=None):
    try:
        vec = iembed.embed_text([text])[0] if kind == "incident" else clip.embed_text([text])[0]
        return [(h.id, h.score, h.payload) for h in vector_db.search(kind, vec.tolist(), k, where, ids)]
    except Exception as e:
        print(f"[hybrid] visual search for {kind} unavailable ({type(e).__name__}: {e})", flush=True)
        return []

def _visual_ft(text, k, where, ids=None):
    from pathlib import Path
    ckpt = Path("models/finetuned/text2person/clip_t2p.pt")
    if not ckpt.exists():
        return _visual("person", text, k, where, ids)
    from src.person import t2p
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "m" not in _ft:
        _ft["m"], _ft["tok"] = t2p.load(str(ckpt), dev); _ft["m"].eval()
    with torch.no_grad():
        f = t2p.txt_feat(_ft["m"], _ft["tok"], [text], dev)
        v = torch.nn.functional.normalize(f.float(), dim=-1)[0].cpu().numpy()
    return [(h.id, h.score, h.payload) for h in vector_db.search("person_ft", v.tolist(), k, where, ids)]

_MOT = {}
RUN_WORDS = {"running", "runs", "run", "jogging", "sprinting"}

def query_motion(kind, text):
    if kind == "person" and set(re.findall(r"[a-z]+", text.lower())) & RUN_WORDS:
        return "running"
    return None

def _motion_ids(kind, label):
    if (kind, label) not in _MOT:
        import sqlite3
        from pathlib import Path
        from src.common.config import PATHS
        db = Path(PATHS["qdrant_path"]).parent / "sqlite" / "meta.db"
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        _MOT[(kind, label)] = [r[0] for r in con.execute(
            "select id from tracks where kind=? and json_extract(attrs_json,'$.motion')=? and coalesce(json_extract(attrs_json,'$.speed'),0)<=3.0", (kind, label))]
        con.close()
    return _MOT[(kind, label)]

def rrf(lists, weights=None, k=60):
    s = {}
    for n, lst in enumerate(lists):
        w = weights[n] if weights else 1.0
        for r, i in enumerate(lst):
            s[i] = s.get(i, 0) + w / (k + r + 1)
    return s

def retrieve(kind, text, k=50, mode="hybrid", where=None, pool=100, w=(1.0, 1.0, 1.0), boost=True):
    ml = query_motion(kind, text) if boost else None
    ids = (_motion_ids(kind, ml) or None) if ml else None
    aset = set(ids) if ids else None
    vis = []
    if mode in ("visual", "hybrid"):
        vis = _visual(kind, text, pool, where, ids)
    elif mode in ("visual_ft", "hybrid_ft"):
        vis = _visual_ft(text, pool, where, ids)
    payload = {i: p for i, _, p in vis}
    lists, ws = [], []
    if vis:
        lists.append([i for i, _, _ in vis]); ws.append(w[0])
    if mode in ("text", "hybrid", "hybrid_ft"):
        q = T.embed([text], query=True)[0]
        lists.append([h.id for h in vector_db.search(f"text_{kind}", q.tolist(), pool, None, ids)]); ws.append(w[1])
        lists.append([i for i, _ in T.bm25_search(kind, text, pool if aset is None else 20000) if aset is None or i in aset][:pool]); ws.append(w[2])
    fused = rrf(lists, ws)
    missing = [i for i in fused if i not in payload]
    if missing:
        for pt in vector_db.client().retrieve(kind, ids=missing, with_payload=True):
            payload[pt.id] = pt.payload
    if boost and kind in ("person", "vehicle"):
        for i in fused:
            if i in payload:
                fused[i] += attr_bonus(kind, text, payload[i])
    ids = sorted(fused, key=fused.get, reverse=True)[:k]
    vs = {i: s for i, s, _ in vis}
    out = []
    for i in ids:
        if i not in payload:
            continue
        p = dict(payload[i])
        if where and any(p.get(a) != b for a, b in where.items()):
            continue
        p["id"], p["score"], p["vis_score"] = i, fused[i], vs.get(i)
        out.append(p)
    return out
