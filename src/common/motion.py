import numpy as np

def _prep(path, fps):
    p = np.array(path, float)
    k = 5
    sm = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")
    x, y, h, w = sm(p[:, 1]), sm(p[:, 2]), sm(p[:, 3]), sm(p[:, 4])
    dur = (p[-1, 0] - p[0, 0]) / fps
    return x, y, h, w, max(dur, 1e-6)

def person_motion(path, fps):
    x, y, h, w, dur = _prep(path, fps)
    hh = float(np.median(h))
    speed = np.hypot(np.diff(x), np.diff(y)).sum() / dur / hh   # body heights per second
    disp = np.hypot(x[-1] - x[0], y[-1] - y[0]) / hh
    n = max(len(h) // 3, 1)
    grow = h[-n:].mean() / h[:n].mean()
    if disp < 0.5: direction = "stationary"
    elif grow > 1.2: direction = "toward camera"
    elif grow < 0.83: direction = "away from camera"
    else: direction = "right" if x[-1] > x[0] else "left"
    state = "standing" if speed < 0.2 else "walking" if speed < 1.3 else "running"
    return {"motion": state, "direction": direction, "loitering": bool(dur >= 5 and disp < 1.0),
            "speed": round(float(speed), 2)}

def vehicle_motion(path, fps):
    x, y, h, w, dur = _prep(path, fps)
    speed = np.hypot(np.diff(x), np.diff(y)).sum() / dur / float(np.median(w))  # widths per second
    out = {"motion": "stopped" if speed < 0.3 else "moving", "turn": "none", "speed": round(float(speed), 2)}
    n = len(x) // 3
    if out["motion"] == "moving" and n >= 2:
        a1 = np.arctan2(y[n] - y[0], x[n] - x[0]); a2 = np.arctan2(y[-1] - y[-n - 1], x[-1] - x[-n - 1])
        d = np.degrees((a2 - a1 + np.pi) % (2 * np.pi) - np.pi)
        out["turn"] = "right" if d > 30 else "left" if d < -30 else "straight"
    return out
