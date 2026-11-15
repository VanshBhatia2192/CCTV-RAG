import json, random, sqlite3
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer
from src.common.config import PATHS

NAME = "openai/clip-vit-base-patch32"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

def load(ckpt=None, dev="cuda"):
    m = CLIPModel.from_pretrained(NAME)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(dev), CLIPTokenizer.from_pretrained(NAME)

def prep(img):
    img = img.convert("RGB").resize((224, 224), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD

def img_feat(m, x):
    return m.visual_projection(m.vision_model(pixel_values=x).pooler_output)

def txt_feat(m, tok, texts, dev):
    t = tok(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
    return m.text_projection(m.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output)

def load_vehicles(db_path=None, seed=42):
    db_path = Path(db_path or PATHS["sqlite"]).resolve()
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro&nolock=1", uri=True)
    except Exception:
        con = sqlite3.connect(str(db_path), timeout=60)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT id, crops_dir, caption, attrs_json FROM tracks WHERE kind='vehicle' AND caption IS NOT NULL").fetchall()
    con.close()
    
    out = []
    for r in rows:
        files = sorted(Path(r["crops_dir"]).glob("*.jpg"))
        if not files:
            continue
        # Pick median frame crop
        mid_file = files[len(files) // 2]
        cap = r["caption"].strip()
        if not cap:
            continue
        out.append({"id": r["id"], "path": str(mid_file), "caps": [cap]})
        
    random.Random(seed).shuffle(out)
    n = len(out)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    
    for i, item in enumerate(out):
        if i < n_tr:
            item["split"] = "train"
        elif i < n_tr + n_va:
            item["split"] = "val"
        else:
            item["split"] = "test"
            
    return out

def metrics(sim, qids, gids):
    order = np.argsort(-sim, axis=1)
    r1 = r5 = r10 = 0
    aps = []
    for i in range(len(qids)):
        hits = np.where(gids[order[i]] == qids[i])[0]
        if not len(hits):
            continue
        r1 += hits[0] < 1
        r5 += hits[0] < 5
        r10 += hits[0] < 10
        aps.append(float(np.mean((np.arange(len(hits)) + 1) / (hits + 1))))
    n = max(len(qids), 1)
    return {"R1": r1 / n, "R5": r5 / n, "R10": r10 / n, "mAP": float(np.mean(aps)) if aps else 0.0}

@torch.no_grad()
def evaluate(m, tok, rows, dev="cuda"):
    m.eval()
    G, T = [], []
    for i in range(0, len(rows), 128):
        x = torch.stack([prep(Image.open(r["path"])) for r in rows[i:i + 128]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            G.append(F.normalize(img_feat(m, x).float(), dim=-1).cpu())
    texts = [c for r in rows for c in r["caps"]]
    qids = np.array([r["id"] for r in rows for _ in r["caps"]])
    for i in range(0, len(texts), 256):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            T.append(F.normalize(txt_feat(m, tok, texts[i:i + 256], dev).float(), dim=-1).cpu())
    sim = (torch.cat(T) @ torch.cat(G).T).numpy()
    return metrics(sim, qids, np.array([r["id"] for r in rows]))

