import cv2, numpy as np

def frames(path, a, b, n=8):
    """n evenly spaced RGB frames between a and b seconds of a chunk file."""
    cap = cv2.VideoCapture(str(path)); out = []
    for t in np.linspace(a, max(b - 0.05, a), n):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if ok:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    while out and len(out) < n:
        out.append(out[-1])
    return out
