"""Fine-tune CLIP ViT-B/32 on CityFlow-NL vehicle crops + natural-language descriptions (symmetric InfoNCE, one crop and one
sentence per track per batch so a batch has no duplicate vehicles). Train on every scene except --val-scene; the held-out scene's
cameras are split in two: the first half selects the best epoch, the second half is only used for the final zero-shot vs
fine-tuned comparison (text -> track retrieval, paired bootstrap over tracks)."""
import argparse, json, math, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


def split_tracks(m, val_scene):
    cams = sorted({v["cam"] for v in m.values() if v["scene"] == val_scene})
    sel_c, test_c = set(cams[:len(cams) // 2]), set(cams[len(cams) // 2:])
    tr, sel, test = [], [], []
    for k, v in m.items():
        if not v["nl"] or not v["crops"]:
            continue
        (tr if v["scene"] != val_scene else sel if v["cam"] in sel_c else test).append(k)
    return tr, sel, test, sorted(sel_c), sorted(test_c)


def ranks(sim, pos):
    """0-based rank of the positive gallery item for each query (strictly-greater count)."""
    return (sim > sim[np.arange(len(sim)), pos][:, None]).sum(1)


def metrics(rk):
    return dict(R1=round(float((rk < 1).mean()) * 100, 1), R5=round(float((rk < 5).mean()) * 100, 1),
                R10=round(float((rk < 10).mean()) * 100, 1), MRR=round(float((1 / (rk + 1)).mean()) * 100, 1))


def paired_ci(ra, rb, groups, B=1000, seed=0):
    """Mean(reciprocal-rank / R@1 of A minus B), resampling tracks (queries of one track stay together)."""
    out = {}
    u = np.unique(groups); rng = np.random.RandomState(seed)
    for name, f in (("MRR", lambda r: 1 / (r + 1)), ("R1", lambda r: (r < 1).astype(float))):
        d = f(ra) - f(rb); s = np.array([d[groups == g].sum() for g in u]); c = np.array([(groups == g).sum() for g in u])
        boots = [s[i].sum() / c[i].sum() for i in (rng.randint(0, len(u), len(u)) for _ in range(B))]
        out[name] = [round(float(d.mean()) * 100, 1), round(float(np.percentile(boots, 2.5)) * 100, 1), round(float(np.percentile(boots, 97.5)) * 100, 1)]
    return out


def main():
    import torch, torch.nn.functional as F
    from PIL import Image
    from src.person import t2p
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/processed/cityflow_crops"); ap.add_argument("--val-scene", default="S02")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch", type=int, default=96); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="models/finetuned/vehicle/clip_vehicle.pt"); ap.add_argument("--report", default="outputs/reports/vehicle_ft.json")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.batch = 2, 8
    dev = "cpu" if a.smoke or not torch.cuda.is_available() else "cuda"
    rng = np.random.RandomState(a.seed); torch.manual_seed(a.seed)
    m = json.load(open(Path(a.crops) / "manifest.json"))
    tr, sel, test, sel_c, test_c = split_tracks(m, a.val_scene)
    if a.smoke:
        tr, sel, test = tr[:24], sel[:12], test[:12]
    print(f"train {len(tr)} tracks (scenes != {a.val_scene}) | select {len(sel)} tracks {sel_c} | test {len(test)} tracks {test_c} | device {dev}", flush=True)

    def load(k):
        return np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)) for p in m[k]["crops"]])
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        keys = tr + sel + test; imgs = dict(zip(keys, ex.map(load, keys)))
    print(f"loaded {sum(len(v) for v in imgs.values())} crops in {time.time() - t0:.0f}s", flush=True)

    model, tok = t2p.load(None, dev); model.train()
    MEAN, STD = t2p.MEAN.to(dev), t2p.STD.to(dev)
    def prep(arr):
        return (torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD

    @torch.no_grad()
    def evaluate(keys):
        model.eval(); G = []
        for k in keys:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                f = t2p.img_feat(model, prep(imgs[k]))
            G.append(F.normalize(F.normalize(f.float(), dim=-1).mean(0), dim=0).cpu())
        texts = [s for k in keys for s in m[k]["nl"]]; pos = np.array([i for i, k in enumerate(keys) for _ in m[k]["nl"]]); T = []
        for i in range(0, len(texts), 256):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                f = t2p.txt_feat(model, tok, texts[i:i + 256], dev)
            T.append(F.normalize(f.float(), dim=-1).cpu())
        model.train()
        return (torch.cat(T) @ torch.stack(G).T).numpy(), pos

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.2, betas=(0.9, 0.98), eps=1e-6)
    steps_per = max(1, len(tr) // a.batch); total = steps_per * a.epochs; warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * s / total)))
    S_sel0, pos_sel = evaluate(sel); S_test0, pos_test = evaluate(test)
    base_sel, base_test = metrics(ranks(S_sel0, pos_sel)), metrics(ranks(S_test0, pos_test))
    print("epoch 0 (zero-shot) | select:", base_sel, "| test:", base_test, flush=True)
    best, best_state, hist, step = base_sel["MRR"], None, [dict(epoch=0, select=base_sel)], 0
    for ep in range(1, a.epochs + 1):
        perm = rng.permutation(len(tr)); loss_sum = 0.0
        for b in range(steps_per):
            ks = [tr[i] for i in perm[b * a.batch:(b + 1) * a.batch]]
            if len(ks) < 2:
                continue
            x = prep(np.stack([imgs[k][rng.randint(len(imgs[k]))] for k in ks])); texts = [m[k]["nl"][rng.randint(len(m[k]["nl"]))] for k in ks]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                fi, ft = t2p.img_feat(model, x), t2p.txt_feat(model, tok, texts, dev)
            fi, ft = F.normalize(fi.float(), dim=-1), F.normalize(ft.float(), dim=-1)
            logits = model.logit_scale.exp().clamp(max=100) * fi @ ft.T; lab = torch.arange(len(ks), device=dev)
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            loss_sum += loss.item(); step += 1
        S, p = evaluate(sel); r = metrics(ranks(S, p)); hist.append(dict(epoch=ep, loss=round(loss_sum / steps_per, 4), select=r))
        print(f"epoch {ep} loss {loss_sum / steps_per:.4f} | select: {r} | {time.time() - t0:.0f}s", flush=True)
        if r["MRR"] > best:
            best = r["MRR"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        print("no epoch beat zero-shot on the selection half; keeping the last weights for the report", flush=True)
    else:
        model.load_state_dict(best_state)
    S_test1, _ = evaluate(test); r0, r1 = ranks(S_test0, pos_test), ranks(S_test1, pos_test)
    final = dict(zero_shot=metrics(r0), fine_tuned=metrics(r1), diff_ft_minus_zs=paired_ci(r1, r0, pos_test), n_queries=int(len(pos_test)), n_tracks=len(test),
                 random_R1=round(100 / len(test), 1))
    print("\nTEST HALF", test_c, "(never used for training or epoch selection):\n", json.dumps(final, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out if not a.smoke else a.out + ".smoke")
    json.dump(dict(args=vars(a), history=hist, final=final, select_cams=sel_c, test_cams=test_c), open(a.report if not a.smoke else a.report + ".smoke", "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
