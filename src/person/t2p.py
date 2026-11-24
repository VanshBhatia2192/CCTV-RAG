import json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer

NAME = "openai/clip-vit-base-patch32"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

def load(ckpt=None, dev="cuda"):
    try:
        m = CLIPModel.from_pretrained(NAME, local_files_only=True)
        tok = CLIPTokenizer.from_pretrained(NAME, local_files_only=True)
    except Exception:
        m = CLIPModel.from_pretrained(NAME)
        tok = CLIPTokenizer.from_pretrained(NAME)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(dev), tok

def prep(img):
    img = img.convert("RGB").resize((224, 224), Image.BICUBIC)   # whole image, no center crop
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD

def img_feat(m, x):
    return m.visual_projection(m.vision_model(pixel_values=x).pooler_output)

def txt_feat(m, tok, texts, dev):
    t = tok(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
    return m.text_projection(m.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output)

def rstp(root="data/raw/rstpreid"):
    root = Path(root)
    js = next(root.rglob("data_captions.json"))
    byname = {f.name: f for f in root.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
    out = []
    for d in json.load(open(js)):
        p = js.parent / d["img_path"]
        if not p.exists():
            p = root / d["img_path"]
        if not p.exists():
            p = byname.get(Path(d["img_path"]).name)
        if p is not None and Path(p).exists():
            out.append({"path": str(p), "id": d["id"], "caps": d["captions"], "split": d["split"]})
    return out

def metrics(sim, qids, gids):
    order = np.argsort(-sim, axis=1); r1 = r5 = r10 = 0; aps = []
    for i in range(len(qids)):
        hits = np.where(gids[order[i]] == qids[i])[0]
        if not len(hits):
            continue
        r1 += hits[0] < 1; r5 += hits[0] < 5; r10 += hits[0] < 10
        aps.append(float(np.mean((np.arange(len(hits)) + 1) / (hits + 1))))
    n = len(qids)
    return {"R1": r1 / n, "R5": r5 / n, "R10": r10 / n, "mAP": float(np.mean(aps))}

@torch.no_grad()
def evaluate(m, tok, rows, dev="cuda"):
    m.eval(); G, T = [], []
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
