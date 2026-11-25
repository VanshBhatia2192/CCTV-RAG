"""Person identities across chunk boundaries, and a gender / age vote over each identity.

The tracker restarts every 10 s, so one person is a chain of tracks. Where a track ends at a chunk boundary and a
track of similar size starts at the same spot when the next chunk begins, they are the same person (same rule as
eval/gait/eval_reid_live.py). Each track's caption has its own gender / age guess. Measured on the live index, the two
halves of such a pair disagree on gender in 4.0% of 4,233 pairs (about 2% per track), and in 1,039 more pairs only one
half has a gender. Voting over the whole chain fills in unknowns, outvotes one-off mistakes, and leaves genuinely
conflicting chains unknown (shown as partial matches rather than the wrong gender).
"""
import json
from collections import Counter, defaultdict
import numpy as np

MAJORITY = 2 / 3         # share of the chain's clear answers needed to decide


def continuation_pairs(rows, min_len=10):
    """rows: tracks with id, start_s, end_s, path_json, source, cs/ce (chunk start / end). -> [(id_a, id_b)]"""
    by = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    pairs = []
    for rs in by.values():
        ends = [r for r in rs if r["ce"] - r["end_s"] < 0.25]
        starts = defaultdict(list)
        for r in rs:
            if r["start_s"] - r["cs"] < 0.25:
                starts[round(r["cs"], 1)].append(r)
        for a in ends:
            pa = json.loads(a["path_json"] or "[]")
            if len(pa) < min_len:
                continue
            la, best = pa[-1], None
            for b in starts.get(round(a["ce"], 1), []):
                pb = json.loads(b["path_json"] or "[]")
                if len(pb) < min_len:
                    continue
                fb = pb[0]
                d = float(np.hypot(la[1] - fb[1], la[2] - fb[2]))
                if d < 0.3 * la[3] and abs(la[3] - fb[3]) < 0.25 * la[3] and (best is None or d < best[0]):
                    best = (d, b["id"])
            if best:
                pairs.append((a["id"], best[1]))
    return pairs


def chains(ids, pairs):
    """Union-find over continuation pairs -> {track id: chain root}."""
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return {i: find(i) for i in ids}


def vote(values):
    """Majority of the clear answers; None if there are none or no value reaches MAJORITY."""
    vals = [v for v in values if v]
    if not vals:
        return None
    v, n = Counter(vals).most_common(1)[0]
    return v if n / len(vals) >= MAJORITY else None


GROUP = {"child": "young", "teen": "young", "adult": "grown", "elderly": "grown"}


def consensus(who, root):
    """who: {id: {"gender", "age"}} per track; root: {id: chain root}. -> who voted over each chain.
    Gender: chain majority. Age: chain majority of the exact label; if that is split (e.g. adult vs elderly), the
    young / grown split queries care about decides, and a track keeps its own label only when it is on that side."""
    members = defaultdict(list)
    for i, r in root.items():
        members[r].append(i)
    out = {}
    for ids in members.values():
        own = {i: who.get(i) or {} for i in ids}
        g = vote([own[i].get("gender") for i in ids])
        a = vote([own[i].get("age") for i in ids])
        grp = None if a else vote([GROUP.get(own[i].get("age")) for i in ids])
        for i in ids:
            age = a or (own[i].get("age") if grp and GROUP.get(own[i].get("age")) == grp else None)
            if g or age:
                out[i] = {"gender": g, "age": age}
    return out


def load(db, who):
    """Chain every person track in the index and return the voted gender / age per track."""
    rows = [dict(r) for r in db.execute(
        "SELECT t.id, t.start_s, t.end_s, t.path_json, ch.source, ch.start_s AS cs, ch.end_s AS ce "
        "FROM tracks t JOIN chunks ch ON ch.id=t.chunk_id WHERE t.kind='person'")]
    root = chains([r["id"] for r in rows], continuation_pairs(rows))
    return consensus(who, root)
