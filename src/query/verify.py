"""Check the top results with the VLM, one yes/no question per planned requirement.

The old re-ranker asked "does this match <whole query>?", which a VLM happily answers yes to for anything red when
asked about "red bluewhale". Asking per requirement ("Is the text or logo 'bluewhale' visible?", "Is the
bluewhale item red?") and requiring every required answer to be yes makes the check strict and explainable.
"""
import torch
from src.query import plan as P, rerank as rr

SUBJ = {"person": "the person", "vehicle": "the vehicle", "incident": "the scene"}
PART = {"upper": "upper-body clothing (top, shirt or jacket)", "lower": "lower-body clothing (pants, shorts or skirt)",
        "vehicle": "vehicle"}


def question(c, kind):
    s = SUBJ.get(kind, "the subject")
    t, v, at = c["type"], c["value"], c.get("applies_to")
    if t == "text":
        return f'Is the text, logo or brand name "{v}" visible on {s} or anything {s} has?'
    if t == "term":
        return f'Does this image show {s} matching "{v}"?'
    if t == "attribute":
        return {"woman": f"Is {s} a woman?", "man": f"Is {s} a man?", "child": f"Is {s} a child?",
                "teen": f"Is {s} a teenager?", "elderly": f"Is {s} an elderly person?"}.get(v, f"Is {s} {v}?")
    if t == "object":
        return (f"Is {s} a {v}?" if kind == "vehicle" else f"Does {s} have, wear or carry a {v}?")
    if t == "action":
        return f"Is {s} {v}?" if kind != "incident" else f"Does the scene show {v}?"
    if t == "color":
        if at in (None, "any"):
            return f"Is {s} wearing or showing something {v}?" if kind == "person" else f"Is {s} {v}?"
        if at == "vehicle" or (at in PART and kind == "incident"):
            return f"Is {s} {v}?" if at == "vehicle" else f"Is there someone wearing {v} {PART[at]}?"
        if at in PART:
            return f"Is the {PART[at]} of {s} {v}?"
        return f"Is the {at} of {s} {v}?" if P._is_noun(at) else f'Is the item with "{at}" on it {v}?'
    return f'Does this match "{v}"?'


def _images(kind, h, db):
    if kind == "incident":
        if "path" not in h:
            r = db.execute("SELECT w.rel_start, w.rel_end, c.path FROM windows w JOIN chunks c ON c.id=w.chunk_id "
                           "WHERE w.id=?", (h["id"],)).fetchone()
            if not r:
                return []
            h = dict(h, rel_start=r[0], rel_end=r[1], path=r[2])
    else:
        r = db.execute("SELECT crops_dir FROM tracks WHERE id=?", (h["id"],)).fetchone()
        if not r or not r[0]:
            return []
        h = dict(h, crops_dir=r[0])
    try:
        return rr._images(kind, h)
    except Exception:
        return []


REJECT = -1.0      # a clear "no" (logit margin below this) rejects; between REJECT and 0 is "unsure", kept unverified


def checkable(c, kind, h):
    """Which requirements the VLM can judge from the crops. Motion/direction come from the tracker, not from three
    still crops, so they are not asked for tracks. Text the caption already reads (captions come from the same VLM at
    a higher resolution) is not re-asked: small text in a crop gets a false "no"."""
    from src.query.constraints import evidence
    if c["type"] == "action" and kind != "incident":
        return False
    if c["type"] in ("text", "term") and evidence(c, kind, h.get("caption") or "", h) >= 1.0:
        return False
    return True


def verify(plan, hits, top=12, db=None):
    """Ask the VLM about the first `top` hits. Adds h["verify"] = {label: margin}, h["verified"] True/False/None.
    A clear "no" on a required question (or a clear "yes" on a negated one) marks the hit match="rejected"."""
    from src.query.constraints import label
    if db is None:
        from src.index.sql_db import connect
        db = connect()
    req = [c for c in plan["constraints"] if c["required"]] or plan["constraints"][:1]
    for h in hits[:top]:
        kind = h.get("kind", plan["kind"])
        qs = [(label(c), c, 1) for c in req if checkable(c, kind, h)] + \
             [("not " + label(c), c, -1) for c in plan["negations"] if checkable(c, kind, h)]
        if not qs:
            h["verified"] = None
            continue
        ims = _images(kind, h, db)
        if not ims:
            h["verified"] = None
            continue
        with torch.no_grad():
            res = {name: round(sign * rr.ask_yes(ims, question(c, kind)), 2) for name, c, sign in qs}
        h["verify"] = res
        h["vlm"] = min(res.values())
        h["verified"] = h["vlm"] > 0 if h["vlm"] > REJECT else False
        if h["vlm"] <= REJECT:
            h["match"] = "rejected"
        elif h["vlm"] <= 0:
            h["verified"] = None              # unsure: keep, unverified
    return hits
