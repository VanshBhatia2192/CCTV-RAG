import argparse, json, math, random
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
from src.person import t2p

class DS(torch.utils.data.Dataset):
    def __init__(s, rows): s.rows = rows
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r = s.rows[i]
        return t2p.prep(Image.open(r["path"])), random.choice(r["caps"]), r["id"]

def main(a):
    dev = "cuda"
    rows = t2p.rstp()
    tr, va, te = [[r for r in rows if r["split"] == s] for s in ("train", "val", "test")]
    print(f"train {len(tr)} val {len(va)} test {len(te)} images", flush=True)
    assert tr and va and te, "a split is empty, check the split names"
    idx = {i: k for k, i in enumerate(sorted({r["id"] for r in tr}, key=str))}
    gb = torch.cuda.mem_get_info()[0] / 1e9
    bs = a.bs or (256 if gb >= 40 else 128 if gb >= 20 else 64)
    print(f"free {gb:.1f} GB -> batch {bs}", flush=True)
    m, tok = t2p.load(None, dev)
    rep = {"batch": bs, "epochs": a.epochs, "lr": a.lr}
    rep["zero_shot_fullimage_test"] = t2p.evaluate(m, tok, te, dev)
    print("zero-shot (full image) test:", rep["zero_shot_fullimage_test"], flush=True)
    def collate(b):
        return torch.stack([x[0] for x in b]), [x[1] for x in b], torch.tensor([idx[x[2]] for x in b])
    dl = torch.utils.data.DataLoader(DS(tr), batch_size=bs, shuffle=True, num_workers=6, drop_last=True,
                                     collate_fn=collate, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl); warm = max(1, min(100, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    out = Path("models/finetuned/text2person"); out.mkdir(parents=True, exist_ok=True)
    best, hist = -1, []
    for ep in range(a.epochs):
        m.train(); tot = 0
        for x, caps, ids in dl:
            x, ids = x.to(dev, non_blocking=True), ids.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fi = t2p.img_feat(m, x); ft = t2p.txt_feat(m, tok, caps, dev)
            fi = F.normalize(fi.float(), dim=-1); ft = F.normalize(ft.float(), dim=-1)
            lg = m.logit_scale.exp().clamp(max=100) * fi @ ft.T
            same = (ids[:, None] == ids[None, :]) & ~torch.eye(len(ids), dtype=torch.bool, device=dev)
            lg = lg.masked_fill(same, -1e4)                      # same person in batch is not a negative
            lab = torch.arange(len(ids), device=dev)
            loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
        v = t2p.evaluate(m, tok, va, dev)
        hist.append({"epoch": ep + 1, "loss": tot / len(dl), **{"val_" + k: x for k, x in v.items()}})
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.3f} val {v}", flush=True)
        if v["R1"] + v["mAP"] > best:
            best = v["R1"] + v["mAP"]; torch.save(m.state_dict(), out / "clip_t2p.pt")
    m.load_state_dict(torch.load(out / "clip_t2p.pt", map_location=dev))
    rep["finetuned_test"] = t2p.evaluate(m, tok, te, dev); rep["history"] = hist
    print("FINE-TUNED test:", rep["finetuned_test"], flush=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    json.dump(rep, open("outputs/reports/finetune_t2p.json", "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=0)
    main(ap.parse_args())
