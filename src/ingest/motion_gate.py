import cv2, numpy as np

def motion_score(path, sample_fps=2, size=(160, 90), pix_thresh=25):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(round(fps / sample_fps)), 1)
    prev, scores, i = None, [], 0
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY), (5, 5), 0)
                if prev is not None:
                    scores.append(float((cv2.absdiff(g, prev) > pix_thresh).mean()))
                prev = g
        i += 1
    cap.release()
    return float(np.mean(scores)) if scores else 0.0

def has_motion(path, threshold=0.02):
    s = motion_score(path)
    return s >= threshold, s
