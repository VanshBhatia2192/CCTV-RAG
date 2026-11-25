"""Score each planned requirement separately against a candidate's caption and attributes.

Plain fusion adds up evidence, so a query like "red bluewhale" ranks anything red highly because "bluewhale"
contributes nothing. Here every required constraint has to be satisfied for a "full" match; results that miss one
are kept only as labelled partial matches, and a required term that no caption in the index mentions at all is
reported as unsupported so the API can say so instead of pretending.
"""
import json, re
from src.query import plan as P

OBJ_SYN = {"bag": {"bag", "backpack", "handbag", "purse", "tote", "satchel", "briefcase"},
           "hat": {"hat", "cap", "beanie"}, "cap": {"cap", "hat", "beanie"},
           "car": {"car", "sedan", "hatchback", "taxi", "coupe"}, "truck": {"truck", "pickup"},
           "van": {"van", "minivan"}, "suv": {"suv"}, "bus": {"bus"},
           "motorcycle": {"motorcycle", "motorbike", "scooter"}, "bicycle": {"bicycle", "bike", "cyclist"},
           "jacket": {"jacket", "coat", "puffer", "parka", "windbreaker"}, "coat": {"coat", "jacket", "parka"},
           "shirt": {"shirt", "t-shirt", "tshirt", "top", "blouse"}, "hoodie": {"hoodie", "hooded", "sweatshirt"},
           "pants": {"pants", "trousers", "jeans", "slacks"}, "jeans": {"jeans", "denim"},
           "phone": {"phone", "cellphone", "smartphone"}, "suitcase": {"suitcase", "luggage", "rolling"},
           "package": {"package", "box", "parcel", "carton", "boxes"}, "box": {"box", "package", "parcel", "carton"},
           "uniform": {"uniform", "police", "security", "sheriff"}, "vest": {"vest", "hi-vis", "reflective"},
           "umbrella": {"umbrella"}}
ATTR_SYN = {"woman": {"woman", "women", "lady", "girl", "female"}, "man": {"man", "men", "guy", "boy", "male"},
            "child": {"child", "children", "kid", "kids", "boy", "girl", "toddler"},
            "teen": {"teen", "teenage", "teenager"}, "elderly": {"elderly", "old", "senior"}}
ACT_SYN = {"walking": {"walk", "walks", "walking"}, "running": {"run", "runs", "running", "jogging", "jogs"},
           "standing": {"standing", "stands", "stationary", "still", "waiting", "idle"},
           "sitting": {"sitting", "sits", "seated"}, "carrying": {"carrying", "carries", "holding", "holds"},
           "parked": {"parked", "stopped", "stationary"}, "driving": {"driving", "drives", "moving"},
           "moving": {"moving", "driving", "walking", "running"}, "turning": {"turning", "turns"},
           "talking": {"talking", "talks", "conversation"}, "using phone": {"phone", "texting"},
           "riding": {"riding", "rides"}, "entering": {"entering", "enters"}, "exiting": {"exiting", "exits",
           "leaving"}, "loading": {"loading", "unloading"}, "crossing": {"crossing", "crosses"}}
# event words in scene captions ("people fighting" -> "two men punch each other", "a scuffle"). Stems; "push" is left
# out on purpose (pushing a cart or stroller, kicking a ball are far more common than a fight).
INCIDENT_STEMS = {"fight": ("fight", "punch", "hitting", "slap", "scuffl", "brawl", "wrestl", "altercat", "assault", "attack"),
                  "fall": ("fall", "fell", "trip", "slip", "collaps"),
                  "crash": ("crash", "collid", "collision", "accident"),
                  "steal": ("steal", "stole", "theft", "thief", "rob", "snatch", "shoplift"),
                  "chase": ("chas", "pursu"), "argue": ("argu", "yell", "shout")}
INCIDENT_GROUP = {"fighting": "fight", "assault": "fight", "falling": "fall", "fell": "fall", "accident": "crash",
                  "collision": "crash", "collide": "crash", "robbery": "steal", "theft": "steal", "stealing": "steal",
                  "chasing": "chase", "argument": "argue"}
MOTION_ATTR = {"walking": {"walking"}, "running": {"running"}, "standing": {"standing", "stopped"},
               "parked": {"stopped", "standing"}, "driving": {"moving"}, "moving": {"moving", "walking", "running"}}
COLOR_SYN = {c: {k for k, v in P.COLORS.items() if v == c} for c in set(P.COLORS.values())}
SLOT_NOUNS = {"upper": P.UPPER | {"top", "shirt", "jacket"}, "lower": P.LOWER, "vehicle": set(P.VEHICLES) | {"pickup"}}
SAT = 0.5          # evidence at or above this counts as satisfied


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def has(ncap, phrase, squashed=None):
    """Is phrase in the normalised caption? Long phrases also match with spaces ignored (bluewhale ~ blue whale)."""
    ph = norm(phrase)
    if not ph:
        return False
    if f" {ph} " in f" {ncap} " or f" {ph}s " in f" {ncap} ":
        return True
    flat = ph.replace(" ", "")
    return len(flat) >= 5 and flat in (squashed if squashed is not None else ncap.replace(" ", ""))


GENDERS = {"man", "woman"}
AGES = {"child", "teen", "adult", "elderly"}


def who_of(desc):
    """Gender and age group of a person track from its structured description (tracks.desc_json):
    the focused second opinion (gender_check / age_check, src/captions/person_attrs.py) when it gave a clear answer,
    else the caption's own fields. None = not known."""
    d = desc or {}
    g = d.get("gender_check") if d.get("gender_check") in GENDERS else d.get("gender")
    a = d.get("age_check") if d.get("age_check") in AGES else d.get("age")
    g = str(g or "").lower(); a = str(a or "").lower()
    return {"gender": g if g in GENDERS else None, "age": a if a in AGES else None}


def _who_from_words(words):
    """Fallback when there is no structured description: gender / age from the caption wording."""
    has = lambda key: any(w in ATTR_SYN[key] for w in words)
    m, w_ = has("man"), has("woman")
    age = "child" if has("child") else "teen" if has("teen") else "elderly" if has("elderly") else None
    if age is None and (m or w_) and not any(x in words for x in ("person", "people")):
        age = "adult"                         # v2 sentences say "A man" / "A woman" only for adults
    return {"gender": "man" if m and not w_ else "woman" if w_ and not m else None, "age": age}


class Corpus:
    """All captions, normalised once; answers "how many captions mention X" and "which ids mention X"."""
    def __init__(self, docs, who=None):              # docs: {kind: {id: caption}}, who: {person id: who_of(...)}
        self.docs = docs
        self.who = who or {}
        self.n = {k: {i: norm(c) for i, c in d.items()} for k, d in docs.items()}
        self.flat = {k: {i: c.replace(" ", "") for i, c in d.items()} for k, d in self.n.items()}
        self.vocab = {}                              # word -> number of captions using it
        for d in self.n.values():
            for c in d.values():
                for t in set(c.split()):
                    self.vocab[t] = self.vocab.get(t, 0) + 1

    def caption(self, kind, i):
        return self.docs.get(kind, {}).get(i)

    def ids_with(self, kind, phrase, limit=None):
        n, f = self.n.get(kind, {}), self.flat.get(kind, {})
        out = [i for i, c in n.items() if has(c, phrase, f[i])]
        return out[:limit] if limit else out

    def satisfying(self, kind, plan, limit=None):
        """Ids whose caption alone satisfies every required constraint and no negation: a structured filter over
        captions, used as an extra candidate list so matches outside the embedding top-k are not missed."""
        req = [c for c in plan["constraints"] if c["required"]]
        if not req:
            return []
        out = []
        for i, cap in self.docs.get(kind, {}).items():
            at = {"who": self.who.get(i)} if kind == "person" else None
            if all(evidence(c, kind, cap, at) >= SAT for c in req) and \
                    not any(evidence(c, kind, cap, None) >= SAT for c in plan["negations"]) and \
                    not conflicts(req, kind, cap, at):
                out.append(i)
                if limit and len(out) >= limit:
                    break
        return out

    def support(self, phrase):
        return {k: len(self.ids_with(k, phrase)) for k in self.n}


_CORPUS = None


def corpus(reload=False):
    global _CORPUS
    if _CORPUS is None or reload:
        from src.index.sql_db import connect
        db = connect()
        docs = {"person": {}, "vehicle": {}, "incident": {}}
        for i, k, c in db.execute("SELECT id, kind, caption FROM tracks WHERE caption IS NOT NULL"):
            docs.setdefault(k, {})[i] = c
        who = {}
        try:
            for i, dj in db.execute("SELECT id, desc_json FROM tracks WHERE kind='person' AND desc_json IS NOT NULL"):
                w = who_of(json.loads(dj))
                if w["gender"] or w["age"]:
                    who[i] = w
        except Exception:                              # index without desc_json (v1 captions only)
            pass
        try:                                           # vote gender / age over each person's chain of tracks
            from src.query import identity
            who = identity.load(db, who)
        except Exception as e:
            print("identity vote skipped:", e)
        try:
            docs["incident"] = {i: c for i, c in db.execute("SELECT id, caption FROM windows WHERE caption IS NOT NULL")}
        except Exception:
            pass
        db.close()
        _CORPUS = Corpus(docs, who)
    return _CORPUS


NOUNS = P.UPPER | P.LOWER | P.HEAD | P.FEET | P.ITEMS | set(P.VEHICLES) | P.PEOPLE | {"pickup", "jeans", "denim"}


def _near(words, colors, targets, span=4):
    """A color word that describes one of the target words: the first noun after it (skipping other colors and
    adjectives, so "red and white striped shirt" works) must be a target. "red jacket and blue jeans" is not red jeans."""
    for i, w in enumerate(words):
        if w not in colors:
            continue
        for x in words[i + 1:i + 1 + span]:
            if x in targets:
                return True
            if x in NOUNS:
                break
    return False


def evidence(c, kind, caption, attrs):
    """0..1 support for one constraint from the caption text (preferred) and the zero-shot attributes."""
    attrs = attrs or {}
    ncap = norm(caption)
    words = ncap.split()
    t, v, at = c["type"], c["value"], c.get("applies_to")
    if t in ("text", "term"):
        return 1.0 if has(ncap, v) else 0.0
    if t == "attribute":
        who = attrs.get("who")
        if v in GENDERS or v in AGES:
            key = "gender" if v in GENDERS else "age"
            if who is not None:            # structured / voted answer: unknown stays unknown (do not fall back to the
                return 1.0 if who.get(key) == v else 0.0     # caption wording a vote just overruled)
            wh = _who_from_words(words)
            if wh.get(key):
                return 1.0 if wh[key] == v else 0.0
        return 1.0 if any(w in ATTR_SYN.get(v, {v}) for w in words) else 0.0
    if t == "object":
        syn = OBJ_SYN.get(v, {v})
        if any(has(ncap, s) for s in syn):
            return 1.0
        if v in OBJ_SYN["bag"] and attrs.get("bag") not in (None, "none"):
            return 0.75 if v == "bag" or attrs.get("bag") == v else 0.4
        if v in ("hat", "cap", "beanie") and attrs.get("hat") == "hat":
            return 0.75
        if kind == "vehicle" and attrs.get("type") == v:
            return 0.75
        return 0.0
    if t == "color":
        cols = COLOR_SYN.get(v, {v})
        if at in (None, "any"):
            if any(w in cols for w in words):
                return 1.0
            slot = ("upper_color", "lower_color", "color")
        else:
            targets = SLOT_NOUNS.get(at) or (set(norm(at).split()) | OBJ_SYN.get(at, set()))
            if _near(words, cols, targets):
                return 1.0
            if at not in SLOT_NOUNS and any(x in targets for x in words):
                return 0.0            # the caption names that item, in another color: trust it over zero-shot attrs
            slot = {"upper": ("upper_color",), "lower": ("lower_color",), "vehicle": ("color",)}.get(at) or \
                (("upper_color",) if at in P.UPPER else ("lower_color",) if at in P.LOWER else ())
        if any(attrs.get(s) == v for s in slot):
            return 0.75
        return 0.3 if any(w in cols for w in words) else 0.0      # the color is there, but on something else
    if t == "action":
        parts = v.split()
        base, dirn = parts[0], (parts[1] if len(parts) > 1 else None)
        if P.INCIDENT.fullmatch(base):
            stems = INCIDENT_STEMS.get(INCIDENT_GROUP.get(base, base), (base[:5],))
            return 1.0 if any(w.startswith(st) for w in words for st in stems) else 0.0
        syn = ACT_SYN.get(base, {base})
        ok = any(w in syn for w in words) or attrs.get("motion") in MOTION_ATTR.get(base, set())
        if base == "turning" and dirn:
            return 1.0 if attrs.get("turn") == dirn or has(ncap, f"turning {dirn}") or has(ncap, f"turns {dirn}") else 0.0
        if not dirn:
            return 1.0 if ok else 0.0
        d_ok = dirn in words or str(attrs.get("direction", "")).startswith(dirn) or attrs.get("turn") == dirn
        return (0.6 if ok else 0.0) + (0.4 if d_ok else 0.0)
    return 0.0


def label(c):
    v = f'"{c["value"]}"' if c["type"] == "text" else c["value"]
    at = c.get("applies_to")
    return f"{v} {at}" if c["type"] == "color" and at not in (None, "any") else v


def unsupported(plan, cp):
    """Required text/term constraints that no caption in the index mentions."""
    return [c for c in plan["constraints"] if c["required"] and c["type"] in ("text", "term")
            and sum(cp.support(c["value"]).values()) == 0]


def _missing_label(c, who):
    """'man (gender not visible)' when the gender / age is simply unknown, so a partial result reads as 'could not
    tell', not as 'wrong'."""
    if c["type"] == "attribute" and (c["value"] in GENDERS or c["value"] in AGES):
        key = "gender" if c["value"] in GENDERS else "age"
        if not (who or {}).get(key):
            return f"{label(c)} ({key} not visible)"
    return label(c)


def conflicts(req, kind, caption, attrs):
    """Required gender / age the result clearly contradicts ("man" asked, a woman shown; "child" asked, an adult shown).
    These results are excluded, not shown as partial: a person whose gender / age is unknown stays partial."""
    if kind != "person":
        return []
    want = [c for c in req if c["type"] == "attribute" and (c["value"] in GENDERS or c["value"] in AGES)]
    if not want:
        return []
    wh = (attrs or {}).get("who") or _who_from_words(norm(caption).split())
    out = []
    for c in want:
        key = "gender" if c["value"] in GENDERS else "age"
        if wh.get(key) and wh[key] != c["value"]:
            out.append(f'{label(c)} (is {wh[key]})')
    return out


def score(plan, hits, cp):
    """Annotate hits with per-constraint evidence and reorder: full matches first, then partial, negations last."""
    req = [c for c in plan["constraints"] if c["required"]]
    soft = [c for c in plan["constraints"] if not c["required"]]
    for h in hits:
        kind = h.get("kind", plan["kind"])
        cap = h.get("caption") or cp.caption(kind, h["id"]) or ""
        h["caption"] = cap
        if kind == "person" and "who" not in h:
            h["who"] = cp.who.get(h["id"])
        ev = {id(c): evidence(c, kind, cap, h) for c in plan["constraints"]}
        h["matched"] = [label(c) for c in plan["constraints"] if ev[id(c)] >= SAT]
        h["missing"] = [_missing_label(c, h.get("who")) for c in req if ev[id(c)] < SAT]
        # negations trust the caption when there is one: the zero-shot attributes say "bag" for most people, so
        # "without a bag" used to exclude every result, including captions that list the outfit and no bag
        h["violates"] = [label(c) for c in plan["negations"] if evidence(c, kind, cap, {} if cap else h) >= SAT] + \
            conflicts(req, kind, cap, h)
        h["coverage"] = round(sum(min(ev[id(c)], 1.0) for c in req) / len(req), 3) if req else 1.0
        h["soft"] = sum(ev[id(c)] for c in soft) / len(soft) if soft else 0.0
        h["match"] = "excluded" if h["violates"] else ("full" if not h["missing"] else "partial")
    tier = {"full": 2, "partial": 1, "excluded": 0}
    top = max((h.get("score") or 0.0 for h in hits), default=0.0) or 1.0
    return sorted(hits, key=lambda h: (tier[h["match"]], h["coverage"] + 0.15 * h["soft"]
                                       + 0.5 * (h.get("score") or 0.0) / top), reverse=True)
