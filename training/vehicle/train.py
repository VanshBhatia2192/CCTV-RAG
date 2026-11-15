import argparse, json, math, random
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
from src.vehicle import t2v

class DS(torch.utils.data.Dataset):
    def __init__(s, rows): s.rows = rows
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r = s.rows[i]
        return t2v.prep(Image.open(r["path"])), random.choice(r["caps"]), r["id"]

def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = t2v.load_vehicles()
    tr, va, te = [[r for r in rows if r["split"] == s] for s in ("train", "val", "test")]
    print(f"Vehicle tracks -> train: {len(tr)}, val: {len(va)}, test: {len(te)} images", flush=True)
    if not (tr and va and te):
        raise SystemExit(f"Empty split: train={len(tr)}, val={len(va)}, test={len(te)}. Ensure vehicle tracks are captioned in SQLite.")
        
    idx = {i: k for k, i in enumerate(sorted({r["id"] for r in tr}, key=str))}
    gb = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 16.0
    bs = a.bs or (256 if gb >= 40 else 128 if gb >= 20 else 64)
    print(f"Memory: {gb:.1f} GB -> batch size {bs}", flush=True)
    
    m, tok = t2v.load(None, dev)
    rep = {"batch": bs, "epochs": a.epochs, "lr": a.lr}
    rep["zero_shot_test"] = t2v.evaluate(m, tok, te, dev)
    print("Zero-shot test:", rep["zero_shot_test"], flush=True)
    
    def collate(b):
        return torch.stack([x[0] for x in b]), [x[1] for x in b], torch.tensor([idx[x[2]] for x in b])
        
    dl = torch.utils.data.DataLoader(DS(tr), batch_size=bs, shuffle=True, num_workers=4, drop_last=True,
                                     collate_fn=collate, pin_memory=True if dev == "cuda" else False)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl)
    warm = max(1, min(100, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    
    out = Path("models/finetuned/text2vehicle")
    out.mkdir(parents=True, exist_ok=True)
    best, hist = -1, []
    
    for ep in range(a.epochs):
        m.train()
        tot = 0
        for x, caps, ids in dl:
            x, ids = x.to(dev, non_blocking=True), ids.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16 if dev == "cuda" else torch.float32):
                fi = t2v.img_feat(m, x)
                ft = t2v.txt_feat(m, tok, caps, dev)
            fi = F.normalize(fi.float(), dim=-1)
            ft = F.normalize(ft.float(), dim=-1)
            lg = m.logit_scale.exp().clamp(max=100) * fi @ ft.T
            same = (ids[:, None] == ids[None, :]) & ~torch.eye(len(ids), dtype=torch.bool, device=dev)
            lg = lg.masked_fill(same, -1e4)
            lab = torch.arange(len(ids), device=dev)
            loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            
        v = t2v.evaluate(m, tok, va, dev)
        hist.append({"epoch": ep + 1, "loss": tot / max(len(dl), 1), **{"val_" + k: val for k, val in v.items()}})
        print(f"Epoch {ep + 1}/{a.epochs} loss {tot / max(len(dl), 1):.3f} val {v}", flush=True)
        if v["R1"] + v["mAP"] > best:
            best = v["R1"] + v["mAP"]
            torch.save(m.state_dict(), out / "clip_t2v.pt")
            
    m.load_state_dict(torch.load(out / "clip_t2v.pt", map_location=dev))
    rep["finetuned_test"] = t2v.evaluate(m, tok, te, dev)
    rep["history"] = hist
    print("FINE-TUNED test:", rep["finetuned_test"], flush=True)
    
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    json.dump(rep, open("outputs/reports/finetune_t2v.json", "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=0)
    main(ap.parse_args())

