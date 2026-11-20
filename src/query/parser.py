import json, re, torch
from src.common.clip import DEV
from src.common.config import MODELS
from src.index.sql_db import connect

SYS = ('Parse a CCTV search query. Reply with JSON only: {"type":"person|vehicle|incident",'
       '"text":"short visual description, no camera or time words","camera":null or string}. '
       'person = people by clothes, items, motion. vehicle = cars, trucks, bikes. '
       'incident = events like fight, accident, fire, robbery, fall.')
VEH = r"\b(car|truck|van|suv|bus|bike|motorcycle|vehicle|sedan|taxi)\b"
INC = r"\b(fight|accident|crash|fire|robbery|theft|fall|fell|assault|explosion|vandal|incident|collision)\w*"
_m = _t = None

def _llm(q):
    global _m, _t
    if _m is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            _t = AutoTokenizer.from_pretrained(MODELS["parser_llm"], local_files_only=True)
            _m = AutoModelForCausalLM.from_pretrained(MODELS["parser_llm"], torch_dtype="auto", local_files_only=True).to(DEV).eval()
        except Exception:
            _t = AutoTokenizer.from_pretrained(MODELS["parser_llm"])
            _m = AutoModelForCausalLM.from_pretrained(MODELS["parser_llm"], torch_dtype="auto").to(DEV).eval()
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    ids = _t.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(DEV)
    with torch.no_grad():
        out = _m.generate(**ids, max_new_tokens=100, do_sample=False)
    txt = _t.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))

def _rules(q):
    PERS = r"\b(person|people|man|men|woman|women|male|female|boy|girl|teenager|kid|child|pedestrian|guy|lady|wearing|wears|carrying|carries|jacket|shirt|jeans|backpack|hoodie|hat)\b"
    kind = "incident" if re.search(INC, q, re.I) else "person" if re.search(PERS, q, re.I) else "vehicle" if re.search(VEH, q, re.I) else "person"
    return {"type": kind, "text": q, "camera": None}

def parse(q, use_llm=True):
    try:
        r = _llm(q) if use_llm else _rules(q)
    except Exception:
        r = _rules(q)
    if r.get("type") not in ("person", "vehicle", "incident"):
        r["type"] = _rules(q)["type"]
    cams = {x[0] for x in connect().execute("SELECT DISTINCT camera FROM chunks").fetchall()}
    r["camera"] = r.get("camera") if r.get("camera") in cams else None
    r["text"] = r.get("text") or q
    return r
