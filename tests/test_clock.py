"""Synchronised clock for cross-camera time comparisons (src/index/clock.py)."""
import datetime as dt
from src.index import clock as CK


def test_meva_wall_clock_from_file_name():
    d, t = CK.clock("data/raw/meva/2018-03-07.17-00-00.17-05-00.school.G336.r13.avi", 12.5)
    assert d == "wall" and t == dt.datetime(2018, 3, 7, 17, 0, 0).timestamp() + 12.5


def test_cityflow_scene_clock_uses_cam_timestamp_offsets(tmp_path):
    root = tmp_path / "AICity22_Track1_MTMC_Tracking"
    (root / "cam_timestamp").mkdir(parents=True)
    (root / "cam_timestamp" / "S02.txt").write_text("c006 0\nc007 0.061\nc008 0.421\n")
    src = root / "validation" / "S02" / "c008" / "vdo.avi"
    src.parent.mkdir(parents=True); src.touch()
    d, t = CK.clock(str(src), 10.0)
    assert d == "scene:S02" and abs(t - 10.421) < 1e-9


def test_footage_without_a_shared_clock_is_not_comparable():
    assert CK.clock("data/raw/somewhere/cam1.mp4", 5.0) is None
    assert CK.gap(("wall", 0, 10), ("scene:S02", 0, 10)) is None
    assert CK.gap(("wall", 0, 10), ("wall", 40, 50)) == 30           # 30 s between the end of one and the start of the other
    assert CK.gap(("wall", 0, 10), ("wall", 5, 20)) == 0             # overlapping
