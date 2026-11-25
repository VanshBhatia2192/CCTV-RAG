"""How often is the caption model wrong about gender and age, and does a dedicated question do better?

Ground truth: RSTPReid test images with two human descriptions each ("The woman is wearing ..."). An image is labelled
man / woman only when both descriptions agree; child when either says boy / girl / child / kid. Images are downscaled
to CCTV size (100 px high, like the live crops) so the model cannot read gender off a sharp photo.

  A. caption  = the gender / age fields of the v2 caption JSON (what the index holds today)
  B. focused  = src.captions.person_attrs.ask(): one short question about gender and age only

    python -m eval.captions.gender_eval --n 600 [--h 100]
"""
import argparse, json, random, re
from pathlib import Path
from PIL import Image
from src.captions import v2
from src.captions import person_attrs as PA

RAW = Path("data/raw")
MAN = re.compile(r"\b(man|men|male|gentleman|guy|boy|he|his|him)\b", re.I)
WOMAN = re.compile(r"\b(woman|women|female|lady|girl|she|her|hers)\b", re.I)
CHILD = re.compile(r"\b(boy|girl|child|kid|kids|children|toddler)\b", re.I)


def gt_gender(caps):
    votes = []
    for c in caps:
        m, w = bool(MAN.search(c)), bool(WOMAN.search(c))
        votes.append("man" if m and not w else "woman" if w and not m else None)
    return votes[0] if votes and all(v == votes[0] for v in votes) and votes[0] else None


def score(pred, gt):
    known = [(p, g) for p, g in zip(pred, gt) if g]
    answered = [(p, g) for p, g in known if p in ("man", "woman")]
    wrong = sum(p != g for p, g in answered)
    return dict(n=len(known), answered=len(answered), unknown=len(known) - len(answered), wrong=wrong,
                error_rate_of_answered=round(100 * wrong / max(1, len(answered)), 1),
                accuracy_overall=round(100 * (len(answered) - wrong) / max(1, len(known)), 1))


def main(a):
    rows = [d for d in json.load(open(RAW / "rstpreid/data_captions.json")) if d["split"] == "test"]
    random.Random(1).shuffle(rows)
    rows = [d for d in rows if gt_gender(d["captions"])][:a.n]
    ims, gt, child = [], [], []
    for d in rows:
        im = Image.open(RAW / "rstpreid/imgs" / d["img_path"]).convert("RGB")
        im = im.resize((max(1, round(im.width * a.h / im.height)), a.h), Image.BICUBIC)
        ims.append(im); gt.append(gt_gender(d["captions"])); child.append(any(CHILD.search(c) for c in d["captions"]))
    big = [v2.crop_box(__import__("numpy").array(im), im.width / 2, im.height / 2, im.height / 1.24, im.width / 1.24, "person") for im in ims]
    cap, foc = [], []
    for i in range(0, len(big), a.batch):
        for j, _, _ in v2.describe("person", [[x] for x in big[i:i + a.batch]]):
            cap.append(((j or {}).get("gender") or "unknown").lower())
        foc += [r["gender"] for r in PA.ask([[x] for x in big[i:i + a.batch]])]
    rep = {"images": len(gt), "height_px": a.h, "caption": score(cap, gt), "focused": score(foc, gt),
           "disagree_caption_vs_focused": sum(c in ("man", "woman") and f in ("man", "woman") and c != f for c, f in zip(cap, foc)),
           "gt_children": sum(child)}
    print(json.dumps(rep, indent=1))
    Path(a.report).write_text(json.dumps(rep, indent=1)); print("wrote", a.report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--h", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--report", default="outputs/reports/gender_eval.json")
    main(ap.parse_args())
