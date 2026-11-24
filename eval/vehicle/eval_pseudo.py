"""Caption-as-query check on OUR vehicle index, visual towers only: zero-shot CLIP ('vehicle') vs fine-tuned CLIP ('vehicle_ft').
Queries are model-written captions of the track's own crops, so this is an in-domain sanity check, NOT ground truth (the text-index rows
would be circular and are not reported)."""
import argparse, json
import numpy as np
from training.vehicle.finetune_clip import metrics, paired_ci


def main():
    import torch, torch.nn.functional as F
    from src.common import clip
    from src.index import vector_db
    from src.index.sql_db import connect
    from src.person import t2p
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="models/finetuned/vehicle/clip_vehicle.pt"); ap.add_argument("--report", default="outputs/reports/vehicle_pseudo.json")
    a = ap.parse_args()
    db = connect(); c = vector_db.client()
    rows = [dict(r) for r in db.execute("SELECT id, caption FROM tracks WHERE kind='vehicle' AND caption IS NOT NULL AND caption != ''")]
    rng = np.random.RandomState(a.seed); pick = [rows[i] for i in rng.choice(len(rows), a.n, replace=False)]
    texts = [r["caption"].split(" Motion:")[0].strip() for r in pick]
    N = c.count("vehicle", exact=True).count; N2 = c.count("vehicle_ft", exact=True).count
    print(f"{a.n} queries | gallery: vehicle {N}, vehicle_ft {N2} | e.g. {texts[0]!r}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, tok = t2p.load(a.ckpt, dev); m.eval()
    with torch.no_grad():
        vft = F.normalize(t2p.txt_feat(m, tok, texts, dev).float(), dim=-1).cpu().numpy()
    vzs = clip.embed_text(texts)

    def rank_all(coll, V, N):
        out = []
        for r, v in zip(pick, V):
            ids = [p.id for p in c.query_points(coll, query=v.tolist(), limit=N, with_payload=False).points]
            out.append(ids.index(r["id"]) if r["id"] in ids else N)
        return np.array(out)
    rz, rf = rank_all("vehicle", vzs, N), rank_all("vehicle_ft", vft, N2)
    groups = np.arange(a.n)
    res = dict(zero_shot=metrics(rz), fine_tuned=metrics(rf), diff_ft_minus_zs=paired_ci(rf, rz, groups), n_queries=a.n, gallery=N,
               random_R1=round(100 / N, 3), note="caption-as-query on our own index; captions are model-written from the same crops (not ground truth)")
    print(json.dumps(res, indent=1)); json.dump(res, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
