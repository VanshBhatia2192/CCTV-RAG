"""Caption generation v2: structured, higher-resolution descriptions of tracks and scenes.

Why v1 captions were weak:
  * track crops are tiny (median 98 px high, 10% under 44 px) and were fed at native size, so Qwen2.5-VL saw about a
    dozen visual tokens per crop; scene windows were four 336 px thumbnails of a 1080p frame;
  * free-text prompts gave vague, inconsistent wording ("dark clothing"), hallucinated details, and motion phrases that
    contradicted the tracker ("walks ... Motion: standing stationary");
  * visible text was rarely transcribed ("a bag with white text").

v2:
  * re-crops the K best frames of each track straight from the chunk video (largest box, not cut by the frame edge,
    spread over the track), pads them for context and upscales so the VLM gets ~250 tokens per crop;
  * scenes get 8 frames at up to 768 px, in time order;
  * the VLM fills a JSON schema (garment-by-garment colors/types, bag, carried items, exact visible text, vehicle
    make/type/features, scene events); a deterministic sentence is built from it, so the wording is consistent and
    every field is also stored for structured filtering;
  * posture comes from the VLM only as "posture"; motion/direction stay with the tracker.

Workers write JSONL (outputs/captions_v2/<kind>/<shard>.jsonl, resumable); src.captions.v2_import loads it into SQLite.
    python -m src.captions.v2 --kind person --shard 0 --nshards 3 [--limit 200] [--ids-file ids.txt]
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

VERSION = "v2"
OUT = Path("outputs/captions_v2")
COLOR_WORDS = ("black, white, gray, silver, red, maroon, blue, navy, light blue, green, olive, yellow, brown, beige, "
               "khaki, orange, pink, purple")

PROMPTS = {
"person": """These are {n} crops of the same person from a surveillance camera (low resolution, enlarged).
Describe only what is clearly visible. Reply with ONE JSON object for the person (not one per image), no other text:
{{"gender": "man|woman|unknown", "age": "child|teen|adult|elderly|unknown",
 "top": {{"type": "t-shirt|shirt|polo|jacket|coat|hoodie|sweater|vest|dress|uniform|unknown", "color": "...", "pattern": "plain|striped|checked|logo|graphic|unknown"}},
 "outer": null or {{"type": "jacket|coat|hoodie|vest|cardigan", "color": "..."}},
 "bottom": {{"type": "jeans|pants|shorts|skirt|leggings|unknown", "color": "..."}},
 "shoes": "e.g. white sneakers, or unknown",
 "headwear": null or "e.g. black cap",
 "bag": null or {{"type": "backpack|handbag|shoulder bag|tote|suitcase|plastic bag", "color": "..."}},
 "carrying": ["objects held in the hands, e.g. phone, cup, box, umbrella"],
 "accessories": ["e.g. glasses, face mask, scarf, lanyard"],
 "visible_text": ["exact words or numbers printed on clothing or bags"],
 "hair": "e.g. long dark hair, or unknown",
 "posture": "walking|standing|sitting|running|bending|riding a bike",
 "other": "one short phrase about something else distinctive on the person (not the background), or leave out"}}
Colors: use simple words ({colors}). Leave out any field you cannot see clearly (do not write "unknown").
Do not identify the person.""",
"vehicle": """These are {n} crops of the same vehicle from a surveillance camera (low resolution, enlarged).
Describe only what is clearly visible. Reply with ONE JSON object for the vehicle (not one per image), no other text:
{{"type": "sedan|hatchback|coupe|suv|pickup truck|van|minivan|box truck|delivery truck|bus|motorcycle|bicycle|other",
 "color": "...", "second_color": null or "...",
 "make": "brand only if clearly identifiable, else unknown", "model": "model only if clearly identifiable, else unknown",
 "visible_text": ["exact company names, words or numbers written on the vehicle"],
 "features": ["e.g. roof rack, tinted windows, ladder on roof, open trunk, logo on door, sunroof, damage"],
 "doing": "driving straight|turning|parked|stopped|reversing",
 "where": "street|intersection|highway|parking lot|driveway|bus stop|building entrance|other",
 "other": "one short phrase about something else distinctive on the vehicle, or leave out"}}
Colors: use simple words ({colors}). Leave out any field you cannot see clearly (do not write "unknown").""",
"window": """These are {n} frames, in time order, from a {dur:.0f}-second surveillance clip.
Describe what happens. Reply with JSON only, no other text:
{{"setting": "one of: parking lot, street, sidewalk, building entrance, hallway, room, gym, bus stop, plaza, other",
 "people": <number of people visible>, "vehicles": <number of vehicles visible>,
 "events": [{{"who": "short description, e.g. a man in a red jacket", "action": "short verb phrase", "target": "object or person acted on, or null"}}] (at most 5 different events, no repeats),
 "summary": "one or two plain sentences"}}
Report concrete actions you can see, for example: getting into or out of a car, opening or closing a car door or trunk,
loading or unloading items, carrying, picking up or putting down an object, handing something to someone, talking,
using a phone, sitting down, standing up, riding a bike, entering or leaving a building, a vehicle starting, stopping,
turning or reversing, running, fighting, falling. If nobody does anything special, say who is walking or standing
where. Do not describe faces.""",
}


# ------------------------------------------------------------------ frames
def _border(cx, cy, h, w, W, H, m=4):
    return cx - w / 2 <= m or cy - h / 2 <= m or cx + w / 2 >= W - m or cy + h / 2 >= H - m


def pick_frames(path, k=4, W=None, H=None):
    """path: [[frame, cx, cy, h, w], ...]. One frame per segment of the track: the largest box not cut by the edge."""
    if not path:
        return []
    segs = np.array_split(np.arange(len(path)), min(k, len(path)))
    out = []
    for s in segs:
        best = max(s, key=lambda i: (not (W and _border(*path[i][1:], W, H)), path[i][3] * path[i][4]))
        out.append(path[best])
    return out


def crop_box(img, cx, cy, h, w, kind, target=448, cap=672):
    """Padded crop around a [cx, cy, h, w] box, enlarged so its long side is about `target` px."""
    H, W = img.shape[:2]
    pad = 0.12 if kind == "person" else 0.08
    ph, pw = h * (1 + 2 * pad), w * (1 + 2 * pad)
    if kind == "person":
        pw = max(pw, ph * 0.5)           # keep some width so arms/bags are not cut
    x0, x1 = int(max(0, cx - pw / 2)), int(min(W, cx + pw / 2))
    y0, y1 = int(max(0, cy - ph / 2)), int(min(H, cy + ph / 2))
    im = Image.fromarray(img[y0:y1, x0:x1])
    s = target / max(im.size)
    s = min(s, cap / max(im.size)) if s < 1 else s
    if abs(s - 1) > 0.05:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    return im


def read_frames(video, idxs):
    import cv2
    cap = cv2.VideoCapture(str(video))
    out = {}
    for f in sorted(set(int(i) for i in idxs)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if ok:
            out[f] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def track_images(row, k=4):
    """row needs path_json, chunk path, kind, crops_dir. Falls back to the saved crops if the video is missing."""
    kind = row["kind"]
    pj = json.loads(row["path_json"] or "[]")
    vid = row.get("chunk_path")
    if pj and vid and Path(vid).exists():
        import cv2
        cap = cv2.VideoCapture(str(vid)); W, H = cap.get(3), cap.get(4); cap.release()
        sel = pick_frames(pj, k, W, H)
        frs = read_frames(vid, [p[0] for p in sel])
        ims = [crop_box(frs[p[0]], *p[1:], kind) for p in sel if p[0] in frs and p[3] > 4 and p[4] > 4]
        if ims:
            return ims
    files = sorted(Path(row["crops_dir"] or "/nonexistent").glob("*.jpg"))
    files = sorted(files, key=lambda f: Image.open(f).size[0] * Image.open(f).size[1], reverse=True)[:k]
    return [crop_box(np.array(Image.open(f).convert("RGB")), *(lambda w, h: (w / 2, h / 2, h / 1.2, w / 1.2))(
        *Image.open(f).size), kind) for f in files]


def window_images(path, rel_start, rel_end, n=8, max_side=768):
    from src.common.video import frames
    out = []
    for f in frames(path, rel_start, rel_end, n):
        im = Image.fromarray(f)
        im.thumbnail((max_side, max_side))
        out.append(im)
    return out


# ------------------------------------------------------------------ model
_M = _P = None


def load(dev="cuda"):
    global _M, _P
    if os.getenv("CAPTION_BACKEND") == "vllm":
        return _vllm()
    if _M is None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from src.common.config import MODELS
        path = MODELS["vlm"]["path"]
        _M = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                                device_map={"": 0} if dev == "cuda" else None).eval()
        _P = AutoProcessor.from_pretrained(path, min_pixels=128 * 28 * 28, max_pixels=640 * 28 * 28)
        _P.tokenizer.padding_side = "left"
    return _M, _P


@torch.no_grad()
def chat_json(system, user, max_new_tokens=300):
    """Text-only turn with the same Qwen2.5-VL model (used as the query planner on GPU)."""
    m, p = load()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = p(text=[text], return_tensors="pt").to(m.device)
    out = m.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    j = parse_json(p.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])
    if j is None:
        raise ValueError("planner reply was not JSON")
    return j


_VLLM = None


def _vllm():
    """vLLM engine for the same checkpoint (CAPTION_BACKEND=vllm, needs the separate ~/envs/vllm environment)."""
    global _VLLM
    if _VLLM is None:
        from vllm import LLM
        from transformers import AutoProcessor
        from src.common.config import MODELS
        path = MODELS["vlm"]["path"]
        llm = LLM(model=path, dtype="bfloat16", max_model_len=8192, limit_mm_per_prompt={"image": 8},
                  gpu_memory_utilization=float(os.getenv("VLLM_GPU_UTIL", "0.45")), max_num_seqs=64,
                  mm_processor_kwargs={"min_pixels": 128 * 28 * 28, "max_pixels": 640 * 28 * 28}, seed=0)
        _VLLM = (llm, AutoProcessor.from_pretrained(path))
    return _VLLM


def _generate_vllm(items, max_new_tokens):
    from vllm import SamplingParams
    llm, p = _vllm()
    reqs = []
    for prompt, ims in items:
        msg = [{"role": "user", "content": [{"type": "image"} for _ in ims] + [{"type": "text", "text": prompt}]}]
        reqs.append({"prompt": p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True),
                     "multi_modal_data": {"image": ims}})
    sp = SamplingParams(temperature=0, max_tokens=max_new_tokens, repetition_penalty=1.05)
    return [o.outputs[0].text.strip() for o in llm.generate(reqs, sp, use_tqdm=False)]


@torch.no_grad()
def generate(items, max_new_tokens=320):
    """items: [(prompt, [PIL images])] -> raw strings."""
    if os.getenv("CAPTION_BACKEND") == "vllm":
        return _generate_vllm(items, max_new_tokens)
    m, p = load()
    texts, imgs = [], []
    for prompt, ims in items:
        msg = [{"role": "user", "content": [{"type": "image"} for _ in ims] + [{"type": "text", "text": prompt}]}]
        texts.append(p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        imgs.extend(ims)
    inp = p(text=texts, images=imgs, padding=True, return_tensors="pt").to(m.device)
    out = m.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return [s.strip() for s in p.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]


def parse_json(s):
    """First JSON object in a reply. Tolerates code fences, trailing commas, and replies that list one object per
    crop and run out of tokens ("[{...}, {...}, {..." -> the first complete object)."""
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        t = m.group(0)
        for cand in (t, re.sub(r",\s*([}\]])", r"\1", t)):
            try:
                j = json.loads(cand)
                return j[0] if isinstance(j, list) and j and isinstance(j[0], dict) else j
            except Exception:
                pass
    dec = json.JSONDecoder()
    for i in (k for k, ch in enumerate(s) if ch == "{"):
        for cand in (s[i:], re.sub(r",\s*([}\]])", r"\1", s[i:])):
            try:
                j, _ = dec.raw_decode(cand)
                if isinstance(j, dict) and len(j) >= 3:
                    return j
            except Exception:
                pass
    return None


# ------------------------------------------------------------------ sentence from fields
_ECHO = re.compile(r"\|| e\.g\.|^e\.g\.|objects held|exact words|exact company|short phrase|short description|"
                   r"leave out|\bunknown\b|or null", re.I)


def _ok(v):
    """A real value: not empty/unknown and not schema text the model copied back (option lists, "e.g." hints)."""
    return isinstance(v, str) and bool(v.strip()) and v.strip().lower() not in ("none", "null", "n/a", "empty") \
        and not _ECHO.search(v)


def _clean_list(v):
    return [x.strip() for x in (v or []) if isinstance(x, str) and _ok(x)] if isinstance(v, list) else []


def _item(d, keys=("color", "pattern", "type")):
    if not isinstance(d, dict):
        return d.strip() if _ok(d) else ""
    parts = [d.get("color"), d.get("pattern") if d.get("pattern") not in ("plain",) else None, d.get("type")]
    return " ".join(x.strip() for x in parts if _ok(x))


def _a(phrase):
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase if phrase else ""


def sentence(kind, j):
    """Consistent caption text built from the JSON fields (colors right next to the garment they describe)."""
    if not isinstance(j, dict):
        return None
    if kind == "person":
        who = {"man": "man", "woman": "woman"}.get(str(j.get("gender", "")).lower(), "person")
        age = str(j.get("age", "")).lower()
        if age in ("child", "teen", "elderly"):
            who = {"child": "child", "teen": f"teenage {who}" if who != "person" else "teenager",
                   "elderly": f"elderly {who}"}[age]
        wear = []
        for x in (_item(j.get("outer")), _item(j.get("top")), _item(j.get("bottom")), j.get("shoes"), j.get("headwear")):
            x = x.strip() if isinstance(x, str) else ""
            # keep real garments only: "a black" (color without type) and repeats are dropped
            if _ok(x) and len(x.split()) > 1 or x in ("jeans", "shorts", "sneakers", "boots", "cap", "hat"):
                if x not in wear:
                    wear.append(x)
        s = f"A {who}" + (f" wearing {', '.join(_a(w) if not w.endswith('s') else w for w in wear)}" if wear else "")
        bag = _item(j.get("bag"))
        carry = ([_a(bag)] if bag else []) + [_a(x) for x in _clean_list(j.get("carrying"))]
        if carry:
            s += f", carrying {' and '.join(carry)}"
        acc = _clean_list(j.get("accessories"))
        if acc:
            s += f", with {' and '.join(acc)}"
        s += "."
        txt = _clean_list(j.get("visible_text"))
        if txt:
            s += " Visible text: " + ", ".join(f'"{t}"' for t in txt) + "."
        if _ok(j.get("hair")):
            s += f" Hair: {j['hair'].strip()}."
        if _ok(j.get("posture")):
            s += f" Posture: {j['posture'].strip()}."
        if _ok(j.get("other")) and not re.search(r"\b(behind|background|car|vehicle|parked|wall|building)\b",
                                                 j["other"], re.I):
            s += f" {j['other'].strip().rstrip('.')}."
        return s
    if kind == "vehicle":
        col = " and ".join(x for x in (j.get("color"), j.get("second_color")) if _ok(x))
        mm = " ".join(x for x in (j.get("make"), j.get("model")) if _ok(x))
        typ = j.get("type") if _ok(j.get("type")) else "vehicle"
        s = _a(" ".join(x for x in (col, mm, typ) if x)).capitalize()
        feats = _clean_list(j.get("features"))
        if feats:
            s += " with " + ", ".join(feats)
        s += "."
        txt = _clean_list(j.get("visible_text"))
        if txt:
            s += " Visible text: " + ", ".join(f'"{t}"' for t in txt) + "."
        doing, where = j.get("doing"), j.get("where")
        if _ok(doing) or _ok(where):
            s += " " + " ".join(x for x in ((doing.strip() if _ok(doing) and "|" not in doing else ""),
                                            (f"on a {where.strip()}" if _ok(where) and "|" not in where else "")) if x
                                ).capitalize() + "."
        if _ok(j.get("other")):
            s += f" {j['other'].strip().rstrip('.')}."
        return s.replace(" .", "") if s.endswith(" .") else s
    if kind == "window":
        s = j.get("summary").strip() if _ok(j.get("summary")) else ""
        s = (s[:1].upper() + s[1:]).rstrip(".") + "." if s else ""
        ev = []
        for e in (j.get("events") or [])[:8]:
            if isinstance(e, dict) and _ok(e.get("who")) and _ok(e.get("action")):
                t = e.get("target") if _ok(e.get("target")) and str(e.get("target")).lower() not in e["action"].lower() else ""
                x = " ".join(y.strip() for y in (e["who"], e["action"], t) if y)
                if x.lower() not in (v.lower() for v in ev):
                    ev.append(x)
        if ev:
            s += (" " if s else "") + "Events: " + "; ".join(ev[:5]) + "."
        st = j.get("setting")
        if _ok(st) and "|" not in st and "," not in st:
            s += f" Setting: {st.strip()}."
        return s or None
    return None


def describe(kind, batch, extra=None):
    """batch: [[PIL images]] (one list per item) -> [(json or None, caption or None, raw)]."""
    extra = extra or [{}] * len(batch)
    items = [(PROMPTS[kind].format(n=len(ims), colors=COLOR_WORDS, **e), ims) for ims, e in zip(batch, extra)]
    raws = generate(items, max_new_tokens=380 if kind == "window" else 300)
    out = []
    for r in raws:
        j = parse_json(r)
        out.append((j, sentence(kind, j), r))
    return out


# ------------------------------------------------------------------ worker
def rows_for(kind, shard, nshards, limit, ids=None):
    from src.index.sql_db import connect
    db = connect()
    if kind == "window":
        q = ("SELECT w.id, w.rel_start, w.rel_end, c.path AS chunk_path FROM windows w JOIN chunks c ON c.id=w.chunk_id "
             "WHERE w.id % ? = ? ORDER BY w.id")
    else:
        q = ("SELECT t.id, t.kind, t.crops_dir, t.path_json, c.path AS chunk_path FROM tracks t JOIN chunks c "
             "ON c.id=t.chunk_id WHERE t.kind=? AND t.id % ? = ? ORDER BY t.id")
    args = (nshards, shard) if kind == "window" else (kind, nshards, shard)
    rows = [dict(r) for r in db.execute(q, args)]
    db.close()
    if ids is not None:
        rows = [r for r in rows if r["id"] in ids]
    return rows[:limit] if limit else rows


def main(a):
    out = OUT / a.kind / f"{a.shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                r = json.loads(line)
                if r.get("json") or not a.retry_failed:     # --retry-failed redoes replies that were not JSON
                    done.add(r["id"])
            except Exception:
                pass
    ids = {int(x) for x in open(a.ids_file).read().split()} if a.ids_file else None
    rows = [r for r in rows_for(a.kind, a.shard, a.nshards, a.limit, ids) if r["id"] not in done]
    print(f"{a.kind} shard {a.shard}/{a.nshards}: {len(rows)} to do ({len(done)} already done)", flush=True)
    if not rows:
        return
    load()
    from concurrent.futures import ThreadPoolExecutor
    bs = a.batch or ((32 if a.kind == "window" else 96) if os.getenv("CAPTION_BACKEND") == "vllm" else
                     (4 if a.kind == "window" else 16))

    def prep(r):
        try:
            if a.kind == "window":
                return window_images(r["chunk_path"], r["rel_start"], r["rel_end"])
            return track_images(r)
        except Exception as e:
            print("prep failed", r["id"], type(e).__name__, e, flush=True)
            return []

    t0, n, bad = time.time(), 0, 0
    from collections import deque
    with ThreadPoolExecutor(6) as ex, open(out, "a") as f:
        batches = [rows[i:i + bs] for i in range(0, len(rows), bs)]
        ahead = deque()                       # at most 3 batches of images prepared ahead (bounded memory)
        def submit(k):
            if k < len(batches):
                ahead.append([ex.submit(prep, r) for r in batches[k]])
        for k in range(3):
            submit(k)
        for k, b in enumerate(batches):
            ims = [x.result() for x in ahead.popleft()]
            submit(k + 3)
            keep = [(r, x) for r, x in zip(b, ims) if x]
            extra = [{"dur": r["rel_end"] - r["rel_start"]} for r, _ in keep] if a.kind == "window" else None
            try:
                res = describe(a.kind, [x for _, x in keep], extra) if keep else []
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                res = []
                for (r, x), e in zip(keep, extra or [None] * len(keep)):
                    try:
                        res += describe(a.kind, [x], [e] if e else None)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache(); res.append((None, None, "OOM"))
            for (r, x), (j, cap, raw) in zip(keep, res):
                bad += j is None
                f.write(json.dumps({"id": r["id"], "kind": a.kind, "version": VERSION, "n_images": len(x),
                                    "json": j, "caption": cap, "raw": None if j else raw}) + "\n")
            f.flush()
            n += len(keep)
            if n and (n // bs) % 10 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(rows)}  {n / el:.2f}/s  parse-fail {bad}  eta {(len(rows) - n) / max(n / el, 1e-6) / 60:.0f} min",
                      flush=True)
    print(f"done {n} in {(time.time() - t0) / 60:.1f} min, parse failures {bad}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", nargs="+", choices=["person", "vehicle", "window"], required=True)
    ap.add_argument("--shard", type=int, default=int(os.getenv("SHARD", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.getenv("NSHARDS", 1)))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--ids-file")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    for kind in args.kind:                     # several kinds in one process: the model is loaded once
        main(argparse.Namespace(**{**vars(args), "kind": kind}))
