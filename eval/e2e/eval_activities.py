import argparse, json, re
from collections import Counter
from pathlib import Path
import numpy as np
from src.index.sql_db import connect
from src.captions import hybrid

NAT = {
 "abandon-package": "a person leaves a bag or package on the ground and walks away",
 "close-trunk": "a person closes the trunk of a car", "close-vehicle-door": "a person closes a car door",
 "embrace-interaction": "two people hugging", "enter-through-structure": "a person enters a building through a door",
 "enter-vehicle": "a person gets into a car", "exit-through-structure": "a person leaves a building through a door",
 "exit-vehicle": "a person gets out of a car", "hand-interaction": "two people shaking hands",
 "heavy-carry": "a person carrying a heavy object", "laptop-interaction": "a person using a laptop",
 "load-vehicle": "a person loads items into a vehicle", "object-transfer": "a person hands an object to another person",
 "open-facility-door": "a person opens the door of a building", "open-trunk": "a person opens the trunk of a car",
 "open-vehicle-door": "a person opens a car door", "people-talking": "two people standing and talking",
 "pick-up-object": "a person picks up an object from the ground", "purchasing": "a person buys something at a counter",
 "read-document": "a person reading a document", "riding": "a person riding a bike or scooter",
 "set-down-object": "a person puts an object down", "sit-down": "a person sits down",
 "stand-up": "a person stands up from sitting", "talk-on-phone": "a person talking on a phone",
 "text-on-phone": "a person texting on a phone", "theft": "a person steals something",
 "unload-vehicle": "a person unloads items from a vehicle", "vehicle-dropsoff-person": "a car stops and a person gets out",
 "vehicle-picksup-person": "a car stops and a person gets in", "vehicle-reversing": "a car reversing",
 "vehicle-starting": "a car starts moving", "vehicle-stopping": "a car comes to a stop",
 "vehicle-turning-left": "a car turning left", "vehicle-turning-right": "a car turning right",
 "vehicle-uturn": "a car making a u-turn"}

def activity(src):
    m = re.match(r"ex\d+-(.+)$", Path(src).stem)
    return m.group(1) if m else None

def main(kinds, modes, out="outputs/reports/eval_activities.json"):
    db = connect()
    cv = {r[0]: r[1] for r in db.execute("SELECT id, source FROM chunks")}
    label = {v: activity(v) for v in set(cv.values()) if activity(v)}
    classes = sorted(set(label.values())); cnt = Counter(label.values()); N = len(label)
    queries = [(a, NAT.get(a, a.replace("-", " "))) for a in classes]
    chance = float(np.mean([cnt[a] / N for a, _ in queries]))
    print(f"{N} labelled videos, {len(classes)} classes, chance R@1 = {chance:.3f}")
    res = {"_chance_R1": chance}
    for kind in kinds:
        for mode in modes:
            key = f"{kind}/{mode}"
            try:
                rs = []
                for a, q in queries:
                    seen = []
                    for h in hybrid.retrieve(kind, q, 300, mode):
                        v = cv.get(h["chunk_id"])
                        if v and v not in seen:
                            seen.append(v)
                    rs.append(next((i + 1 for i, v in enumerate(seen) if label.get(v) == a), 0))
                rs = np.array(rs)
                res[key] = {"R@1": float(np.mean((rs > 0) & (rs <= 1))), "R@5": float(np.mean((rs > 0) & (rs <= 5))),
                            "R@10": float(np.mean((rs > 0) & (rs <= 10))),
                            "MRR": float(np.mean([1 / r if r else 0 for r in rs]))}
            except Exception as e:
                res[key] = f"n/a ({type(e).__name__}: {e})"
            print(key, res[key])
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["incident", "person", "vehicle"])
    ap.add_argument("--modes", nargs="+", default=["visual", "text", "hybrid"])
    ap.add_argument("--out", default="outputs/reports/eval_activities.json")
    a = ap.parse_args()
    main(a.kinds, a.modes, a.out)
