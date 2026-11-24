"""Load v2 captions (outputs/captions_v2/<kind>/*.jsonl) into the SQLite index, then rebuild the caption text index.

Reversible: the first run copies meta.db to meta.db.bak-v1 and keeps every old caption in a caption_v1 column;
`--revert` puts the v1 captions back. The structured fields go into desc_json. The tracker's motion phrase is
re-appended the way v1 did ("Motion: walking left."), since search uses it.

Run on the login node with the demo stopped (Qdrant must be up for the text index):
    QDRANT_URL=http://localhost:6335 python -m src.captions.v2_import [--kinds person vehicle window] [--no-text-index]
"""
import argparse, json, shutil, sqlite3
from pathlib import Path
from src.common.config import PATHS
from src.captions.v2 import OUT, parse_json, sentence


def _cols(db, table):
    return {r[1] for r in db.execute(f"PRAGMA table_info({table})")}


def ensure_columns(db):
    for table in ("tracks", "windows"):
        have = _cols(db, table)
        for col in ("caption_v1", "desc_json"):
            if col not in have:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
        db.execute(f"UPDATE {table} SET caption_v1 = caption WHERE caption_v1 IS NULL AND caption IS NOT NULL")
    db.commit()


def motion_suffix(attrs_json):
    a = json.loads(attrs_json or "{}")
    return f" Motion: {a.get('motion', '')} {a.get('direction', a.get('turn', ''))}.".replace(" .", ".") if a else ""


def load(kind):
    recs = {}
    for f in sorted((OUT / kind).glob("*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("json") and r.get("raw"):          # replies the worker could not parse: retry with the
                r["json"] = parse_json(r["raw"])               # current, more tolerant parser
                r["caption"] = sentence(kind, r["json"]) if r["json"] else None
            if r.get("json"):                                  # always rebuild the sentence from the fields, so
                r["caption"] = sentence(kind, r["json"]) or r.get("caption")   # template fixes need no re-captioning
            if r.get("caption"):
                recs[r["id"]] = r                              # failures keep their v1 caption
    return recs


def main(a):
    dbp = Path(PATHS["sqlite"])
    bak = dbp.with_name(dbp.name + ".bak-v1")
    if not bak.exists() and not a.revert:
        src = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True); dst = sqlite3.connect(bak)
        src.backup(dst); dst.close(); src.close()
        print("backup:", bak)
    db = sqlite3.connect(dbp, timeout=120)
    ensure_columns(db)
    if a.revert:
        for table in ("tracks", "windows"):
            db.execute(f"UPDATE {table} SET caption = caption_v1 WHERE caption_v1 IS NOT NULL")
        db.commit(); print("reverted to v1 captions")
    else:
        for kind in a.kinds:
            recs = load(kind)
            if kind == "window":
                # scenes keep the v1 sentence in front of the v2 description: on the 36 MEVA activity classes over all
                # 11k windows, v1+v2 retrieves better (MRR 0.178) than either alone (v1 0.138, v2 0.128)
                v1 = dict(db.execute("SELECT id, caption_v1 FROM windows").fetchall())
                rows = [(f"{v1.get(i) or ''} {r['caption'] or sentence(kind, r['json'])}".strip() if a.combine_windows
                         else r["caption"] or sentence(kind, r["json"]), json.dumps(r["json"]), i) for i, r in recs.items()]
                db.executemany("UPDATE windows SET caption=?, desc_json=? WHERE id=?", rows)
            else:
                attrs = dict(db.execute("SELECT id, attrs_json FROM tracks WHERE kind=?", (kind,)).fetchall())
                rows = [((r["caption"] or sentence(kind, r["json"])) + motion_suffix(attrs.get(i)), json.dumps(r["json"]), i)
                        for i, r in recs.items() if i in attrs]
                db.executemany("UPDATE tracks SET caption=?, desc_json=? WHERE id=?", rows)
            db.commit()
            total = db.execute("SELECT count(*) FROM windows" if kind == "window" else
                               "SELECT count(*) FROM tracks WHERE kind=?", () if kind == "window" else (kind,)).fetchone()[0]
            print(f"{kind}: {len(rows)} of {total} captions replaced with v2")
    db.close()
    if not a.no_text_index:
        from src.captions import text_index
        text_index.build()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["person", "vehicle", "window"])
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--no-text-index", action="store_true")
    ap.add_argument("--v2-only-windows", dest="combine_windows", action="store_false",
                    help="scene captions = v2 only (default: v1 sentence + v2 description)")
    main(ap.parse_args())
