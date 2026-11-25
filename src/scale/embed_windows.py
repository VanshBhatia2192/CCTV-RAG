import argparse
import torch
from tqdm import tqdm
from src.common.config import MODELS
from src.common.video import frames
from src.incident import embed as iembed
from src.index import vector_db
from src.index.sql_db import connect

class DS(torch.utils.data.Dataset):
    def __init__(s, rows): s.rows, s.p = rows, None
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        if s.p is None:
            from transformers import XCLIPProcessor
            s.p = XCLIPProcessor.from_pretrained(MODELS["text_video"]["path"])
        r = s.rows[i]
        frs = frames(r["path"], r["rel_start"], r["rel_end"], 8)
        if len(frs) < 8:
            return None
        return r["id"], s.p(videos=[list(frs)], return_tensors="pt")["pixel_values"][0]

def collate(b):
    return [x for x in b if x is not None]

def main(limit, workers):
    db = connect()
    rows = [dict(r) for r in db.execute("SELECT w.*, c.path, c.camera, coalesce(c.dataset, 'meva') AS dataset FROM windows w JOIN chunks c ON c.id=w.chunk_id "
                                        "WHERE w.embedded=0 LIMIT ?", (limit,)).fetchall()]
    print(len(rows), "windows to embed", flush=True)
    if not rows:
        return
    by = {r["id"]: r for r in rows}
    m, _ = iembed._load()
    dl = torch.utils.data.DataLoader(DS(rows), batch_size=16, num_workers=workers, collate_fn=collate)
    for batch in tqdm(dl, total=(len(rows) + 15) // 16, mininterval=30):
        if not batch:
            continue
        ids = [b[0] for b in batch]
        pv = torch.stack([b[1] for b in batch]).to(iembed.DEV)
        with torch.no_grad():
            v = iembed._t(m.get_video_features(pixel_values=pv)).cpu().numpy()
        pays = [{"window_id": i, "chunk_id": by[i]["chunk_id"], "camera": by[i]["camera"], "dataset": by[i]["dataset"], "path": by[i]["path"],
                 "rel_start": by[i]["rel_start"], "rel_end": by[i]["rel_end"], "start_s": by[i]["start_s"],
                 "end_s": by[i]["end_s"]} for i in ids]
        vector_db.upsert("incident", ids, v.tolist(), pays)
        db.executemany("UPDATE windows SET embedded=1 WHERE id=?", [(i,) for i in ids]); db.commit()
    print("windows embedded", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10**7); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); main(a.limit, a.workers)
