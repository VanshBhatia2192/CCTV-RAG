"""Re-read an indexed chunk video and cut per-frame person crops for windows of consecutive track frames."""
import cv2, numpy as np

MIN_N = 30   # tracks shorter than this are skipped (same gate as the MEVID eval)
W = 64

def windows(n, K):
    if n < MIN_N:
        return []
    if n <= W:
        return [[0, n]]
    k = min(K, n // W)
    return [[int(s), int(s) + W] for s in np.linspace(0, n - W, k).round()]

def xyxy(pt, fmt, Wv, Hv):
    """pt = [frame, x, y, a, b]. fmt: cxcywh | tlwh | cxcyhw | tlhw (c=centre, tl=top-left; wh/hw = order of a,b)."""
    _, x, y, a, b = pt
    w, h = (b, a) if fmt.endswith("hw") else (a, b)
    x1, y1 = (x - w / 2, y - h / 2) if fmt.startswith("cxcy") else (x, y)
    return max(0, int(round(x1))), max(0, int(round(y1))), min(Wv, int(round(x1 + w))), min(Hv, int(round(y1 + h)))

def read_crops(vpath, pjs, K, fmt, min_wh=(8, 16)):
    """pjs: list of path_json lists (one per track). Returns {(track_idx, window_idx): [BGR crop, ...]}."""
    cap = cv2.VideoCapture(vpath)
    Wv, Hv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    want, out = {}, {}
    for ti, pj in enumerate(pjs):
        for wi, (lo, hi) in enumerate(windows(len(pj), K)):
            out[(ti, wi)] = []
            for pos in range(lo, hi):
                want.setdefault(int(pj[pos][0]), []).append((ti, wi, xyxy(pj[pos], fmt, Wv, Hv)))
    f, last = 0, max(want) if want else -1
    while f <= last:
        if f in want:
            ok, fr = cap.read()
            if not ok:
                break
            for ti, wi, (x1, y1, x2, y2) in want[f]:
                if x2 - x1 >= min_wh[0] and y2 - y1 >= min_wh[1]:
                    out[(ti, wi)].append(fr[y1:y2, x1:x2].copy())
        elif not cap.grab():
            break
        f += 1
    cap.release()
    return out
