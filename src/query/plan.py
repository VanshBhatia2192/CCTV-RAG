"""Query planner: turn a free-text search into a structured plan of separate requirements.

Example: "red bluewhale" ->
    kind "any", constraints [text "bluewhale" (required), color "red" applied to "bluewhale"]
instead of the bag of words {red, bluewhale} that plain embedding/BM25 matching sees.

The rules planner is the default (no model, runs in milliseconds on the CPU demo node). The LLM planner
(use_llm=True) fills the same schema and is checked against the same vocabulary; if it fails, the rules plan
is used. Corpus support (does any caption mention this term at all?) is added by src.query.constraints.
"""
import json, re

COLORS = {"black": "black", "white": "white", "gray": "gray", "grey": "gray", "silver": "gray", "red": "red",
          "maroon": "red", "burgundy": "red", "crimson": "red", "blue": "blue", "navy": "blue", "teal": "blue",
          "green": "green", "olive": "green", "yellow": "yellow", "gold": "yellow", "brown": "brown", "tan": "brown",
          "beige": "brown", "khaki": "brown", "orange": "orange", "pink": "pink", "purple": "purple", "violet": "purple",
          "dark": "dark", "light": "light"}
UPPER = {"jacket", "shirt", "top", "hoodie", "coat", "tshirt", "t-shirt", "sweater", "sweatshirt", "vest", "dress",
         "blouse", "jersey", "puffer", "uniform", "cardigan", "parka", "windbreaker", "hood"}
LOWER = {"jeans", "pants", "trousers", "shorts", "skirt", "leggings", "joggers"}
HEAD = {"hat", "cap", "beanie", "helmet", "headscarf", "scarf"}
FEET = {"shoes", "sneakers", "boots", "sandals"}
ITEMS = {"backpack", "bag", "handbag", "purse", "suitcase", "luggage", "umbrella", "phone", "box", "cart",
         "stroller", "sign", "cup", "bottle", "laptop", "package", "parcel", "tote", "briefcase", "glasses",
         "sunglasses", "mask", "logo", "text"}
VEHICLES = {"car": "car", "sedan": "car", "hatchback": "car", "suv": "suv", "van": "van", "minivan": "van",
            "truck": "truck", "pickup": "truck", "lorry": "truck", "bus": "bus", "motorcycle": "motorcycle",
            "motorbike": "motorcycle", "scooter": "motorcycle", "bicycle": "bicycle", "bike": "bicycle",
            "taxi": "car", "cab": "car", "vehicle": None, "vehicles": None, "cars": "car", "trucks": "truck"}
PEOPLE = {"person", "people", "man", "men", "woman", "women", "guy", "guys", "lady", "ladies", "boy", "boys", "girl",
          "girls", "kid", "kids", "child", "children", "pedestrian", "pedestrians", "someone", "student", "students",
          "male", "female", "adult", "teenager", "worker", "officer", "group"}
# surface form -> canonical action
ACTIONS = {"walk": "walking", "walks": "walking", "walking": "walking", "walked": "walking",
           "run": "running", "runs": "running", "running": "running", "ran": "running", "jogging": "running",
           "sprinting": "running", "stand": "standing", "stands": "standing", "standing": "standing",
           "stood": "standing", "waiting": "standing", "idle": "standing", "loitering": "standing",
           "sit": "sitting", "sits": "sitting", "sitting": "sitting", "seated": "sitting",
           "carry": "carrying", "carries": "carrying", "carrying": "carrying", "holding": "carrying",
           "talk": "talking", "talking": "talking", "phone-call": "talking", "texting": "using phone",
           "park": "parked", "parked": "parked", "parking": "parked", "stopped": "parked", "stationary": "parked",
           "moving": "moving", "drive": "driving", "drives": "driving", "driving": "driving",
           "turn": "turning", "turns": "turning", "turning": "turning", "riding": "riding", "rides": "riding",
           "enter": "entering", "enters": "entering", "entering": "entering", "exit": "exiting",
           "exits": "exiting", "exiting": "exiting", "leaving": "exiting", "loading": "loading",
           "unloading": "loading", "waving": "waving", "pushing": "pushing", "pulling": "pulling",
           "crossing": "crossing", "crosses": "crossing", "opening": "opening", "closing": "closing"}
DIRECTIONS = {"left", "right", "away", "toward", "towards"}
INCIDENT = re.compile(r"\b(fight|fighting|accident|crash|fire|robbery|theft|steal|stealing|fall|fell|falling|"
                      r"assault|explosion|vandal\w*|incident|collision|collide|argument|chase|chasing)\b")
STOP = {"a", "an", "the", "with", "in", "on", "at", "of", "and", "or", "to", "from", "wearing", "wears", "wear",
        "dressed", "who", "is", "are", "was", "were", "that", "which", "near", "by", "show", "me", "find", "search",
        "for", "any", "all", "some", "something", "footage", "video", "videos", "clip", "clips", "camera", "cameras",
        "cam", "looking", "look", "like", "has", "have", "having", "his", "her", "their", "its", "into", "out",
        "over", "under", "this", "these", "those", "there", "where", "when", "get", "getting", "got", "please",
        "one", "two", "it", "be", "being", "very", "big", "small", "large", "little", "while", "colored",
        "coloured", "color", "colour", "clothes", "clothing", "outfit", "side", "front", "back", "up", "down"}
NEG = {"without", "no", "not", "never", "except"}

KINDS = ("person", "vehicle", "incident", "any")
TYPES = ("color", "object", "action", "text", "term")


def _c(type_, value, applies_to=None, required=True, note=None):
    d = {"type": type_, "value": value, "applies_to": applies_to, "required": required}
    if note:
        d["note"] = note
    return d


def _is_noun(w):
    return w in UPPER or w in LOWER or w in HEAD or w in FEET or w in ITEMS or w in VEHICLES or w in PEOPLE


def _plain(t):
    return t not in STOP and t not in COLORS and t not in NEG and not _is_noun(t) and t not in ACTIONS and \
        t not in DIRECTIONS and not INCIDENT.fullmatch(t)


def _slot(noun):
    """What a color describes: the garment/item itself (so "red jacket" is not satisfied by a red shirt under a
    black jacket), or the whole vehicle for vehicle nouns."""
    return "vehicle" if noun in VEHICLES else noun


RARE_DF = 50        # a caption word used by at most this many captions is distinctive enough to require


def is_rare(phrase, vocab):
    """vocab may be a set (no frequencies: treat as common) or a {word: number of captions} mapping."""
    if not isinstance(vocab, dict):
        return False
    return any(vocab.get(t, 0) <= RARE_DF for t in tokens(phrase))


def osa(a, b):
    """Edit distance counting a swap of two neighbouring letters as one edit."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


_LEX = None


def spell_fix(word):
    """Closest word of the search vocabulary (colors, garments, items, vehicles, people, actions) within 1 edit
    (2 for long words). Only domain words: correcting toward arbitrary caption words turns "whale" into "while"."""
    global _LEX
    if _LEX is None:
        _LEX = set(COLORS) | UPPER | LOWER | HEAD | FEET | ITEMS | set(VEHICLES) | PEOPLE | set(ACTIONS)
    lim = 1 if len(word) < 8 else 2
    best = min(((osa(word, v), v) for v in _LEX if abs(len(v) - len(word)) <= lim), default=None)
    return best[1] if best and best[0] <= lim and len(best[1]) >= 3 else None


def tokens(q):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", q.lower())


def rules_plan(q, cameras=(), vocab=None):
    """Deterministic planner. vocab: set of words seen in captions (enables typo fixes and unknown-term detection)."""
    raw = q.strip()
    low = raw.lower()
    cons, negs, notes = [], [], []
    camera = next((c for c in sorted(cameras, key=len, reverse=True) if c and c.lower() in low), None)
    if camera:
        low = low.replace(camera.lower(), " ")
    for m in re.finditer(r'"([^"]+)"|“([^”]+)”', low):           # quoted -> visible text / logo
        cons.append(_c("text", (m.group(1) or m.group(2)).strip()))
    low = re.sub(r'"[^"]*"|“[^”]*”', " ", low)

    w = tokens(low)
    kind_votes = {"person": 0, "vehicle": 0}
    neg_left = 0                      # words still under a "without/no/not"
    pending_color = []                # colors waiting for the noun they describe
    i = 0
    while i < len(w):
        x = w[i]
        if x in NEG:
            neg_left = 3; i += 1; continue
        target = negs if neg_left > 0 else cons
        neg_left = max(0, neg_left - 1)
        if x in COLORS:
            pending_color.append(COLORS[x]); i += 1; continue
        if x in STOP or x.isdigit():   # colors stay pending across "and": "red and white jacket"
            i += 1; continue
        if x in ACTIONS:
            act = ACTIONS[x]
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if nxt in DIRECTIONS:
                act = f"{act} {'toward' if nxt.startswith('toward') else nxt}"; i += 1
            target.append(_c("action", act))
            if act.startswith(("driving", "parked", "turning")): kind_votes["vehicle"] += 1
            i += 1; continue
        if x in DIRECTIONS:
            target.append(_c("action", f"moving {x}", required=False)); i += 1; continue
        if INCIDENT.fullmatch(x):
            target.append(_c("action", x)); i += 1; continue
        noun = x
        if not _is_noun(noun) and noun not in COLORS and (vocab is None or noun not in vocab) and len(noun) >= 3:
            fix = spell_fix(noun)
            if fix:
                notes.append(f'read "{noun}" as "{fix}"'); noun = fix
                if noun in COLORS:
                    pending_color.append(COLORS[noun]); i += 1; continue
                if noun in ACTIONS:
                    w[i] = noun; continue          # re-run this word through the action branch
        if noun in PEOPLE:
            kind_votes["person"] += 2
            for col in pending_color: target.append(_c("color", col, "any"))
            pending_color = []; i += 1; continue
        if _is_noun(noun):
            if noun in VEHICLES:
                kind_votes["vehicle"] += 2
                val = VEHICLES[noun]
                if val: target.append(_c("object", val))
            else:
                kind_votes["person"] += 1
                if noun not in ("logo", "text"): target.append(_c("object", noun))
            for col in pending_color: target.append(_c("color", col, _slot(noun)))
            pending_color = []; i += 1; continue
        # anything else: a descriptive word, a name, a brand or visible text. Glue consecutive unknown words.
        w[i] = noun
        j = i
        while vocab is not None and j + 1 < len(w) and _plain(w[j + 1]) and w[j] not in vocab and w[j + 1] not in vocab:
            j += 1                    # "golden dragon" style names: consecutive words captions never use
        phrase = " ".join(w[i:j + 1])
        known = vocab is not None and all(t in vocab for t in phrase.split())
        # common caption words are soft preferences; rare ones ("whitman", "budget") and words captions never use
        # (names, brands, text: "bluewhale") are required
        if not known:
            target.append(_c("text", phrase))
        else:
            target.append(_c("term", phrase, required=is_rare(phrase, vocab)))
        nxt = w[j + 1] if j + 1 < len(w) else ""
        if not _is_noun(nxt):         # "yellow school bus": the color waits for "bus"
            for col in pending_color: target.append(_c("color", col, phrase))
            pending_color = []
        i = j + 1
    for col in pending_color:          # trailing colors: "the car was red"
        cons.append(_c("color", col, "vehicle" if kind_votes["vehicle"] > kind_votes["person"] else "any"))

    if INCIDENT.search(low) or any(c["type"] == "action" and INCIDENT.fullmatch(c["value"].split()[0]) for c in cons):
        kind = "incident"
    elif kind_votes["vehicle"] > kind_votes["person"]:
        kind = "vehicle"
    elif kind_votes["person"] > 0:
        kind = "person"
    else:
        kind = "any"
    return finalize({"kind": kind, "text": _visual_text(raw, camera, negs), "constraints": cons, "negations": negs,
                     "camera": camera, "notes": notes, "planner": "rules"})


def _visual_text(raw, camera, negs=()):
    """Text for the embedding/BM25 search: the query minus camera names and negated phrases (embedding
    "without a bag" pulls in bags)."""
    t = raw.replace('"', " ")
    t = re.sub(r"\b(without|no|not|except)\s+(any\s+|a\s+|an\s+|the\s+)?[\w-]+", " ", t, flags=re.I) \
        if negs else t
    if camera:
        t = re.sub(re.escape(camera), " ", t, flags=re.I)
    t = re.sub(r"\b(show me|find|search for|footage of|video of|clips? of|on camera|cameras?)\b", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip() or raw


def finalize(p):
    """Validate/normalise any plan (rules or LLM) into the schema; drop duplicates and junk."""
    out = {"kind": p.get("kind") if p.get("kind") in KINDS else "any",
           "text": (p.get("text") or "").strip(), "camera": p.get("camera"),
           "notes": list(p.get("notes") or []), "planner": p.get("planner", "rules")}
    for key in ("constraints", "negations"):
        seen, lst = set(), []
        for c in p.get(key) or []:
            if not isinstance(c, dict) or c.get("type") not in TYPES or not str(c.get("value") or "").strip():
                continue
            c = _c(c["type"], str(c["value"]).strip().lower(), c.get("applies_to") and str(c["applies_to"]).lower(),
                   bool(c.get("required", True)), c.get("note"))
            if c["type"] == "color":
                c["value"] = COLORS.get(c["value"], c["value"])
            sig = (c["type"], c["value"], c["applies_to"])
            if sig not in seen:
                seen.add(sig); lst.append(c)
        out[key] = lst
    return out


def describe(p):
    """Short human-readable summary of what the planner understood, e.g. for the UI."""
    parts = []
    for c in p["constraints"]:
        v = f'"{c["value"]}"' if c["type"] == "text" else c["value"]
        parts.append(f'{v} {c["applies_to"]}' if c["type"] == "color" and c["applies_to"] not in (None, "any")
                     else v)
    parts += [f'not {c["value"]}' for c in p["negations"]]
    who = {"any": "anything", "incident": "event"}.get(p["kind"], p["kind"])
    return f'{who}: ' + ", ".join(parts) if parts else who


# ---------------------------------------------------------------- LLM planner (same schema)
SYS = """You turn a CCTV video search query into JSON. Reply with JSON only, no prose:
{"kind": "person|vehicle|incident|any",
 "text": "short visual description to embed, no camera or time words",
 "constraints": [{"type": "color|object|action|text|term", "value": "...", "applies_to": null or "upper|lower|vehicle|<noun or text it describes>", "required": true}],
 "negations": [same shape, for things that must NOT be present],
 "camera": null}
Rules: one constraint per separate requirement. "text" type = a brand, logo, name or words written on clothing,
bags, vehicles or signs (e.g. "budget", "fedex"). If a word is not ordinary English, treat it as text.
Colors must say what they describe in applies_to. kind is "any" when the query could be a person or a vehicle.
Example: "red bluewhale" -> {"kind":"any","text":"red bluewhale","constraints":[{"type":"text","value":"bluewhale","applies_to":null,"required":true},{"type":"color","value":"red","applies_to":"bluewhale","required":true}],"negations":[],"camera":null}
Example: "woman in a black coat without a bag walking left" -> {"kind":"person","text":"woman in a black coat walking left","constraints":[{"type":"color","value":"black","applies_to":"upper","required":true},{"type":"object","value":"coat","applies_to":null,"required":true},{"type":"action","value":"walking left","applies_to":null,"required":true}],"negations":[{"type":"object","value":"bag","applies_to":null,"required":true}],"camera":null}"""


def llm_plan(q, cameras=(), vocab=None):
    from src.query import parser            # reuses its lazily loaded parser LLM
    p = parser.chat_json(SYS, q, max_new_tokens=220)
    p["planner"] = "llm"
    if p.get("camera") not in set(cameras):
        p["camera"] = next((c for c in cameras if c and c.lower() in q.lower()), None)
    p = finalize(p)
    if vocab is not None:                    # the LLM can't know our captions: re-type words captions never use
        for c in p["constraints"]:
            if c["type"] == "term" and not all(t in vocab for t in tokens(c["value"])):
                c["type"] = "text"
    if not p["text"] or p["negations"]:
        p["text"] = _visual_text(q, p["camera"], p["negations"])
    return p


def merge(llm, rules):
    """A small LLM drops requirements ("white truck turning right" -> just "white vehicle") and keeps typos
    ("blu hoodie"). Keep its plan, fix values with the domain spell-checker, and add back every rules requirement
    it has no equivalent for."""
    def fix(v):
        return " ".join(spell_fix(t) or t if t not in COLORS and not _is_noun(t) and t not in ACTIONS else t
                        for t in tokens(v)) or v
    for key in ("constraints", "negations"):
        for c in llm[key]:
            if c["type"] in ("color", "object", "action"):
                c["value"] = fix(c["value"])
                if c["type"] == "color" and " " in c["value"]:        # "blue hoodie" -> blue on hoodie
                    col, _, rest = c["value"].partition(" ")
                    if col in COLORS:
                        c["value"], c["applies_to"] = COLORS[col], c["applies_to"] or rest
        have = {(c["type"], c["value"]) for c in llm[key]}
        llm[key] += [c for c in rules[key] if (c["type"], c["value"]) not in have]
    if llm["kind"] == "any" and rules["kind"] != "any":
        llm["kind"] = rules["kind"]
    llm["camera"] = llm["camera"] or rules["camera"]
    llm["notes"] += rules["notes"]
    return finalize(llm)


def make_plan(q, use_llm=False, cameras=(), vocab=None):
    rules = rules_plan(q, cameras, vocab)
    if not use_llm:
        return rules
    try:
        p = llm_plan(q, cameras, vocab)
        if not p["constraints"] and rules["constraints"]:
            raise ValueError("empty LLM plan")
        return merge(p, rules)
    except Exception as e:
        rules["notes"].append(f"LLM planner unavailable ({type(e).__name__}); used rules")
        return rules


if __name__ == "__main__":
    import sys
    print(json.dumps(make_plan(" ".join(sys.argv[1:]) or "red bluewhale"), indent=1))
