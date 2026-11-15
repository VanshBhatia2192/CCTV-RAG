import argparse
import torch
from PIL import Image
from tqdm import tqdm
from src.captions.caption_tracks import load_vlm
from src.common.video import frames
from src.index.sql_db import connect

PROMPT = ("These are frames from a surveillance clip in time order. Describe what happens in one or two sentences. "
          "Mention people, vehicles, their actions, and any unusual incident. Do not describe faces.")

class DS(torch.utils.data.Dataset):
    def __init__(s, rows): s.rows = rows
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r, ims = s.rows[i], []
        for f in frames(r["path"], r["rel_start"], r["rel_end"], 8)[::2][:4]:
            im = Image.fromarray(f); im.thumbnail((336, 336)); ims.append(im)
        return r["id"], ims

def ident(b):
    return b

@torch.no_grad()
def describe(m, p, items):
    texts, imgs = [], []
    for _, ims in items:
        msg = [{"role": "user", "content": [{"type": "image"} for _ in ims] + [{"type": "text", "text": PROMPT}]}]
        texts.append(p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)); imgs.extend(ims)
    inp = p(text=texts, images=imgs, padding=True, return_tensors="pt").to(m.device)
    out = m.generate(**inp, max_new_tokens=70, do_sample=False)
    return [s.strip() for s in p.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]

def safe(m, p, items):
    try:
        return describe(m, p, items)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(items) == 1:
            return [None]
        h = len(items) // 2
        return safe(m, p, items[:h]) + safe(m, p, items[h:])

def main(limit):
    db = connect()
    rows = [dict(r) for r in db.execute("SELECT w.id, w.rel_start, w.rel_end, c.path FROM windows w JOIN chunks c "
                                        "ON c.id=w.chunk_id WHERE w.caption IS NULL LIMIT ?", (limit,)).fetchall()]
    print(len(rows), "windows to caption", flush=True)
    if not rows:
        return
    m, p, bs = load_vlm()
    dl = torch.utils.data.DataLoader(DS(rows), batch_size=bs, num_workers=4, collate_fn=ident)
    for batch in tqdm(dl, total=(len(rows) + bs - 1) // bs, mininterval=30):
        batch = [b for b in batch if b[1]]
        if not batch:
            continue
        for (wid, _), cap in zip(batch, safe(m, p, batch)):
            if cap:
                db.execute("UPDATE windows SET caption=? WHERE id=?", (cap, wid))
        db.commit()
    print("window captions done", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=10**7)
    main(ap.parse_args().limit)
