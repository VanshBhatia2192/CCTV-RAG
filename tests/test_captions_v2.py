"""Caption v2 pure functions: JSON parsing, sentence building, frame selection, cropping (no model needed)."""
import numpy as np
import pytest

pytest.importorskip("torch")
from src.captions import v2


def test_parse_json_tolerates_fences_and_trailing_commas():
    assert v2.parse_json('```json\n{"type": "suv", "color": "white",}\n```') == {"type": "suv", "color": "white"}
    assert v2.parse_json("sure! {\"a\": [1, 2,]} done") == {"a": [1, 2]}
    assert v2.parse_json("no json here") is None
    # one object per crop, cut off by the token limit: keep the first complete one
    cut = '```json\n[\n {"gender": "man", "age": "adult", "top": {"type": "shirt", "color": "red"}},\n {"gender": "man", "age": "ad'
    assert v2.parse_json(cut) == {"gender": "man", "age": "adult", "top": {"type": "shirt", "color": "red"}}


def test_person_sentence_keeps_colors_next_to_garments_and_drops_junk():
    j = {"gender": "woman", "top": {"type": "jacket", "color": "black"}, "outer": {"type": "jacket", "color": "black"},
         "bottom": {"type": "skirt", "color": "black"}, "shoes": "black", "headwear": "black cap",
         "bag": {"type": "backpack", "color": "red"}, "carrying": ["phone", "unknown"], "visible_text": ["NIKE"],
         "posture": "walking", "other": "car parked behind"}
    s = v2.sentence("person", j)
    assert s.startswith("A woman wearing a black jacket, a black skirt, a black cap")
    assert s.count("black jacket") == 1 and "a black," not in s and "behind" not in s
    assert "carrying a red backpack and a phone" in s and 'Visible text: "NIKE"' in s and "unknown" not in s


def test_schema_text_copied_by_the_model_is_dropped():
    j = {"gender": "man", "top": {"type": "jacket", "color": "red"}, "posture": "walking|standing|sitting",
         "carrying": ["objects held in the hands, e.g. phone, cup", "phone"], "hair": "unknown"}
    s = v2.sentence("person", j)
    assert s == "A man wearing a red jacket, carrying a phone."
    assert v2.sentence("vehicle", {"type": "sedan|hatchback|coupe", "color": "white"}) == "A white vehicle."


def test_vehicle_and_window_sentences():
    v = v2.sentence("vehicle", {"type": "pickup truck", "color": "white", "make": "ford", "model": "unknown",
                                "visible_text": ["Budget"], "features": ["ladder on roof"], "doing": "parked",
                                "where": "parking lot"})
    assert v == 'A white ford pickup truck with ladder on roof. Visible text: "Budget". Parked on a parking lot.'
    w = v2.sentence("window", {"setting": "parking lot|street", "summary": "a man gets into a car",
                               "events": [{"who": "a man", "action": "opens the door of", "target": "a car"}] * 6})
    assert w == "A man gets into a car. Events: a man opens the door of a car."


def test_pick_frames_prefers_big_boxes_away_from_the_edge():
    path = [[f, 100, 100, 50, 20] for f in range(10)]
    path[3] = [3, 100, 100, 120, 60]           # biggest in the first half
    path[7] = [7, 5, 100, 300, 200]            # biggest overall but cut by the left edge
    path[8] = [8, 500, 300, 90, 40]
    sel = [p[0] for p in v2.pick_frames(path, k=2, W=1920, H=1080)]
    assert sel == [3, 8]


def test_crop_box_enlarges_small_people():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    im = v2.crop_box(img, 500, 500, 90, 30, "person")
    assert max(im.size) == 448 and im.size[0] >= im.size[1] * 0.45
