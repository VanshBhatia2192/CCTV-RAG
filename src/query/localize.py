import numpy as np
from src.common.video import frames
from src.incident import embed as iembed

def localize(kind, text_vec, c, chunk):
    dur = chunk["end_s"] - chunk["start_s"]
    if kind != "incident":  # track times are already exact
        a = max(c["start_s"] - chunk["start_s"] - 0.5, 0)
        b = min(c["end_s"] - chunk["start_s"] + 0.5, dur)
        return a, max(b, a + 1)
    best, span = -1, (c["rel_start"], c["rel_end"])
    s = c["rel_start"]
    while s + 4 <= c["rel_end"] + 1e-6:  # 4 s sub-windows, 1 s step
        frs = frames(c["path"], s, s + 4, 8)
        if len(frs) == 8:
            sc = float(iembed.embed_video(frs) @ text_vec)
            if sc > best:
                best, span = sc, (s, s + 4)
        s += 1
    return span
