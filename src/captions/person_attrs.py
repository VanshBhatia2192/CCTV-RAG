"""A focused second opinion on gender and age for person tracks.

The v2 caption asks for gender among a dozen other fields from small CCTV crops, and sometimes gets it wrong. Here the
same model (Qwen2.5-VL) is asked only about gender and age, from up to four enlarged crops of the track, and may
answer "unclear". eval/captions/gender_eval.py measures both against human-labelled RSTPReid images.

Run over the index (one GPU, vLLM env recommended; resumable, one JSON line per track):
    python -m src.captions.person_attrs --shard 0 --nshards 1      -> outputs/person_attrs/0.jsonl
Then load into the index:
    python -m src.captions.person_attrs --import
"""
import argparse, json, re
from pathlib import Path
from src.captions import v2

OUT = Path("outputs/person_attrs")
PROMPT = """These {n} images show ONE person from a security camera.
Look only at this person. Answer with ONE JSON object and nothing else:
{{"gender": "man|woman|unclear", "age": "child|teen|adult|elderly|unclear"}}
Use "unclear" when the images do not show it clearly (back view, very small, covered). Do not guess."""
G = {"man": "man", "male": "man", "boy": "man", "woman": "woman", "female": "woman", "girl": "woman"}
A = {"child": "child", "kid": "child", "teen": "teen", "teenager": "teen", "adult": "adult", "elderly": "elderly", "old": "elderly"}


def parse(raw):
    j = v2.parse_json(raw) or {}
    g = str(j.get("gender", "")).lower().strip()
    a = str(j.get("age", "")).lower().strip()
    if not j:                                     # tolerate a bare word reply
        w = re.findall(r"[a-z]+", (raw or "").lower())
        g = next((x for x in w if x in G), "")
        a = next((x for x in w if x in A), "")
    return {"gender": G.get(g, "unclear"), "age": A.get(a, "unclear")}


def ask(batch):
    """batch: [[PIL images]] -> [{"gender": man|woman|unclear, "age": child|teen|adult|elderly|unclear}]"""
    items = [(PROMPT.format(n=len(ims)), ims) for ims in batch]
    return [parse(r) for r in v2.generate(items, max_new_tokens=40)]


def run(a):
    out = OUT / f"{a.shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out) if l.strip()}
    rows = [r for r in v2.rows_for("person", a.shard, a.nshards, a.limit) if r["id"] not in done]
    print(f"{len(rows)} person tracks to check (shard {a.shard}/{a.nshards}, {len(done)} already done)", flush=True)
    with open(out, "a") as f:
        for i in range(0, len(rows), a.batch):
            blk = rows[i:i + a.batch]
            ims = [v2.track_images(r) for r in blk]
            keep = [(r, x) for r, x in zip(blk, ims) if x]
            for (r, _), res in zip(keep, ask([x for _, x in keep]) if keep else []):
                f.write(json.dumps({"id": r["id"], **res}) + "\n")
            f.flush()
            if (i // a.batch) % 20 == 0:
                print(f"  {i + len(blk)}/{len(rows)}", flush=True)
    print("done", flush=True)


def load_results():
    res = {}
    for f in sorted(OUT.glob("*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line); res[r["id"]] = r
    return res


def import_results():
    """Store the focused answers in tracks.desc_json as "gender_check" / "age_check" (the caption fields stay as they
    were, so the effect can be compared or undone)."""
    from src.index.sql_db import connect
    res = load_results()
    db = connect()
    n = 0
    for tid, dj in db.execute("SELECT id, desc_json FROM tracks WHERE kind='person'").fetchall():
        r = res.get(tid)
        if not r:
            continue
        d = json.loads(dj or "{}")
        d["gender_check"], d["age_check"] = r["gender"], r["age"]
        db.execute("UPDATE tracks SET desc_json=? WHERE id=?", (json.dumps(d), tid)); n += 1
    db.commit()
    print(f"stored focused gender/age for {n} person tracks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--import", dest="do_import", action="store_true")
    a = ap.parse_args()
    import_results() if a.do_import else run(a)
