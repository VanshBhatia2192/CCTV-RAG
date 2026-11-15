import argparse, json, sqlite3
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from src.common.config import MODELS
from src.index.sql_db import connect

PROMPTS = {
 "person": "These are frames of the same person at different moments in one surveillance video. Treat them as one person and do not mention multiple images or angles. In one sentence of at most 25 words, describe the clothing (colors and types), carried items and posture. Do not describe the face or guess identity.",
 "vehicle": "These are frames of the same vehicle at different moments in one surveillance video. Treat them as one vehicle and do not mention multiple images or angles. In one sentence of at most 20 words, give the vehicle type, color and any visible detail."}

def load_vlm():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    free, _ = torch.cuda.mem_get_info()
    gb = free / 1e9
    kw = {"device_map": {"": 0}}
    if gb >= 30: mode, bs = "bf16", 8
    elif gb >= 20: mode, bs = "bf16", 4
    elif gb >= 10: mode, bs = "4bit", 4
    else: raise SystemExit(f"only {gb:.1f} GB free on this GPU, need at least 10")
    if mode == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = torch.bfloat16
    print(f"free GPU memory {gb:.1f} GB -> {mode}, batch {bs}", flush=True)
    path = MODELS["vlm"]["path"]
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kw).eval()
    p = AutoProcessor.from_pretrained(path)
    p.tokenizer.padding_side = "left"
    return m, p, bs

@torch.no_grad()
def describe_batch(m, p, items):
    texts, imgs = [], []
    for r, ims in items:
        msg = [{"role": "user", "content": [{"type": "image"} for _ in ims] + [{"type": "text", "text": PROMPTS[r["kind"]]}]}]
        texts.append(p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        imgs.extend(ims)
    inp = p(text=texts, images=imgs, padding=True, return_tensors="pt").to(m.device)
    out = m.generate(**inp, max_new_tokens=60, do_sample=False)
    return [s.strip() for s in p.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]

def main(kind, limit):
    db = connect()
    try:
        db.execute("ALTER TABLE tracks ADD COLUMN caption TEXT"); db.commit()
    except sqlite3.OperationalError:
        pass
    kinds = ("person", "vehicle") if kind == "all" else (kind,)
    q = "SELECT id, kind, crops_dir, attrs_json FROM tracks WHERE caption IS NULL AND kind IN (%s) LIMIT ?" % ",".join("?" * len(kinds))
    rows = db.execute(q, (*kinds, limit)).fetchall()
    print(len(rows), "tracks to caption", flush=True)
    if not rows:
        return
    m, p, bs = load_vlm()
    i = done = skipped = 0
    bar = tqdm(total=len(rows), desc="captions")
    while i < len(rows):
        batch = rows[i:i + bs]
        items = []
        for r in batch:
            ims = []
            files = sorted(Path(r["crops_dir"]).glob("*.jpg"))
            for f in files[:: max(len(files) // 3, 1)][:3]:
                im = Image.open(f).convert("RGB"); im.thumbnail((336, 336)); ims.append(im)
            if ims:
                items.append((r, ims))
        try:
            caps = describe_batch(m, p, items) if items else []
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1:
                bs = max(1, bs // 2); print("OOM -> batch", bs, flush=True); continue
            skipped += len(batch); i += len(batch); bar.update(len(batch)); continue
        for (r, _), cap in zip(items, caps):
            a = json.loads(r["attrs_json"] or "{}")
            extra = f" Motion: {a.get('motion', '')} {a.get('direction', a.get('turn', ''))}." if a else ""
            db.execute("UPDATE tracks SET caption=? WHERE id=?", (cap + extra, r["id"]))
        db.commit(); done += len(items)
        i += len(batch); bar.update(len(batch))
    print(f"captioned {done}, skipped {skipped}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["person", "vehicle", "all"], default="all")
    ap.add_argument("--limit", type=int, default=10**6)
    a = ap.parse_args()
    main(a.kind, a.limit)
