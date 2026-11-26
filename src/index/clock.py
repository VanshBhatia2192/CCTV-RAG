"""A synchronised clock for footage, so sightings on different cameras can be compared in time.

clock(source, offset_s) -> (domain, seconds) or None
  * MEVA-style file names with a timestamp (2018-03-07.17-00-00....): ("wall", epoch seconds)
  * CityFlow / AI City layout (.../S05/c010/vdo.avi with cam_timestamp/S05.txt next to the scene folders):
    ("scene:S05", per-camera offset + position in the video), the official synchronisation
  * anything else: None (no shared clock; time cannot be compared across cameras)
Two sightings are comparable only when their domains are equal.
"""
import datetime as dt
import re
from functools import lru_cache
from pathlib import Path

_STAMP = re.compile(r"(20\d\d)-(\d\d)-(\d\d)\.(\d\d)-(\d\d)-(\d\d)")
_CITYFLOW = re.compile(r"(S\d\d)[/\\](c\d\d\d)[/\\]vdo\.\w+$")


@lru_cache(maxsize=None)
def _offsets(root, scene):
    f = Path(root) / "cam_timestamp" / f"{scene}.txt"
    if not f.exists():
        return {}
    out = {}
    for line in open(f):
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = float(parts[1])
    return out


def _cityflow_root(source):
    for p in Path(source).parents:
        if (p / "cam_timestamp").is_dir():
            return str(p)
    return None


def clock(source, offset_s=0.0):
    source = str(source or "")
    m = _STAMP.search(source)
    if m:
        return "wall", dt.datetime(*map(int, m.groups())).timestamp() + float(offset_s or 0.0)
    m = _CITYFLOW.search(source)
    if m:
        root = _cityflow_root(source)
        if root:
            scene, cam = m.groups()
            return f"scene:{scene}", _offsets(root, scene).get(cam, 0.0) + float(offset_s or 0.0)
    return None


def gap(a, b):
    """Seconds between two intervals (domain, start, end) on the same clock; None if the clocks differ."""
    if a is None or b is None or a[0] != b[0]:
        return None
    return max(b[1] - a[2], a[1] - b[2], 0.0)
