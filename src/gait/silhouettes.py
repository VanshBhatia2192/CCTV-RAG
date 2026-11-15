import random, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path("data/processed/mevid_gait")
SEG = "models/pretrained/yolov8m-seg.pt"

def norm64(mask):
    """uint8 0/255 HxW -> 64x64 uint8 silhouette (rows cropped to mask, height 64, centred on mass)."""
    ys = np.where(mask.any(1))[0]
    if len(ys) < 20:
        return None
    img = mask[ys[0]:ys[-1] + 1]
    w = max(1, int(round(64 * img.shape[1] / img.shape[0])))
    img = cv2.resize(img, (w, 64), interpolation=cv2.INTER_CUBIC)
    col = img.sum(0, dtype=np.float64).cumsum()
    if col[-1] == 0:
        return None
    xc = int(np.searchsorted(col, col[-1] / 2))
    img = np.pad(img, ((0, 0), (32, 32)))
    return img[:, xc:xc + 64].astype(np.uint8)

def person_mask(model, im):
    r = model.predict(im, imgsz=320, classes=[0], retina_masks=True, verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None
    d = r.masks.data.cpu().numpy()
    k = int(d.reshape(len(d), -1).sum(1).argmax())
    return (d[k] > 0.5).astype(np.uint8) * 255

def fit(img, s=128):
    h, w = img.shape[:2]; f = s / max(h, w)
    img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((s, s, 3), np.uint8); out[:img.shape[0], :img.shape[1]] = img
    return out

def sample(n=12):
    from ultralytics import YOLO
    random.seed(0)
    files = [str(p) for p in (ROOT / "test").glob("*/*.jpg")]
    print("frames extracted so far:", len(files))
    sel = random.sample(files, min(n, len(files)))
    model = YOLO(SEG); strips, sizes, miss = [], [], 0
    for f in sel:
        im = cv2.imread(f); sizes.append(im.shape[:2])
        m = person_mask(model, im); s64 = norm64(m) if m is not None else None
        if s64 is None:
            miss += 1; strips.append(np.hstack([fit(im), np.zeros((128, 256, 3), np.uint8)])); continue
        ov = im.copy(); ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        strips.append(np.hstack([fit(im), fit(ov), fit(cv2.cvtColor(s64, cv2.COLOR_GRAY2BGR))]))
    rows = [np.hstack(strips[i:i + 2]) for i in range(0, len(strips) - 1, 2)]
    out = Path("outputs/gait_prep/sil_sample.png"); cv2.imwrite(str(out), np.vstack(rows))
    hs, ws = zip(*sizes)
    print(f"no-mask/too-small: {miss}/{len(sel)} | crop height median {int(np.median(hs))} (min {min(hs)}), width median {int(np.median(ws))}")
    print("wrote", out)

if __name__ == "__main__":
    sample()
