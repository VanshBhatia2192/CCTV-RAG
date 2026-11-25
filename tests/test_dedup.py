"""Folding repeated sightings into one result (src/query/dedup.py), with stub vectors instead of the index."""
import numpy as np
from src.query import dedup

A = np.array([1.0, 0.0, 0.0])                      # person A
A2 = np.array([0.97, 0.243, 0.0])                  # person A, another track (cos 0.97)
B = np.array([0.0, 1.0, 0.0])                      # someone else


def card(i, cam="cam1", t="2018-03-07T12:00:00", kind="person", ds="meva", dur=10.0):
    return dict(id=i, kind=kind, dataset=ds, camera=cam, recorded_at=t, start_s=0.0, end_s=dur)


def run(cards, vecs, monkeypatch):
    monkeypatch.setattr(dedup, "_vectors", lambda kind, ids: {i: v / np.linalg.norm(v) for i, v in vecs.items() if i in ids})
    return dedup.group(cards)


def test_same_person_same_camera_minutes_apart_folds(monkeypatch):
    out = run([card(1), card(2, t="2018-03-07T12:02:00"), card(3, t="2018-03-07T12:00:30")], {1: A, 2: A2, 3: B}, monkeypatch)
    assert [c["id"] for c in out] == [1, 3]
    assert out[0]["n_sightings"] == 2 and out[0]["sightings"][0]["id"] == 2


def test_same_look_but_far_apart_in_time_stays_separate(monkeypatch):
    out = run([card(1), card(2, t="2018-03-07T12:30:00")], {1: A, 2: A2}, monkeypatch)
    assert len(out) == 2


def test_two_people_visible_at_the_same_time_never_fold(monkeypatch):
    out = run([card(1, dur=20.0), card(2, t="2018-03-07T12:00:05")], {1: A, 2: A}, monkeypatch)
    assert len(out) == 2


def test_same_camera_needs_spatial_continuity(monkeypatch):
    monkeypatch.setattr(dedup, "_vectors", lambda kind, ids: {i: A for i in ids})
    cs = [card(1), card(2, t="2018-03-07T12:01:00")]
    near = dedup.group(cs, meta={1: {"last": (100, 200, 120)}, 2: {"first": (130, 210, 118)}})
    far = dedup.group(cs, meta={1: {"last": (100, 200, 120)}, 2: {"first": (900, 210, 118)}})
    assert len(near) == 1 and len(far) == 2


def test_vehicles_are_not_folded_across_cameras(monkeypatch):
    out = run([card(1, kind="vehicle"), card(2, cam="cam2", kind="vehicle", t="2018-03-07T12:00:03")], {1: A, 2: A}, monkeypatch)
    assert len(out) == 2


def test_overlapping_cameras_fold_only_within_seconds(monkeypatch):
    same_moment = run([card(1), card(2, cam="cam2", t="2018-03-07T12:00:05")], {1: A, 2: A}, monkeypatch)
    later = run([card(1), card(2, cam="cam2", t="2018-03-07T12:02:00")], {1: A, 2: A}, monkeypatch)
    assert len(same_moment) == 1 and len(later) == 2


def test_never_folds_across_datasets_or_kinds(monkeypatch):
    out = run([card(1), card(2, ds="aicity"), card(3, kind="vehicle")], {1: A, 2: A, 3: A}, monkeypatch)
    assert len(out) == 3


def test_untimestamped_footage_uses_position_in_the_same_video(monkeypatch):
    a = dict(card(1, t=None), start_s=10.0, end_s=20.0); b = dict(card(2, t=None), start_s=40.0, end_s=50.0)
    c = dict(card(3, t=None), start_s=40.0, end_s=50.0)
    monkeypatch.setattr(dedup, "_vectors", lambda kind, ids: {i: A for i in ids})
    out = dedup.group([a, b, c], meta={1: {"source": "v1.avi"}, 2: {"source": "v1.avi"}, 3: {"source": "v2.avi"}})
    assert [x["id"] for x in out] == [1, 3]           # 1 and 2 share a video and are 30 s apart; 3 is another video
