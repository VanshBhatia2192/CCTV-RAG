"""Query planning and per-requirement scoring (no models, no index needed)."""
from src.query import plan as P, constraints as C

DOCS = {
    "person": {
        1: "The person wears a red hoodie, black pants, and carries a backpack. Motion: walking left.",
        2: "The person wears a blue t-shirt with \"Whitman Football\" text, black pants. Motion: standing stationary.",
        3: "The person wears a red jacket and blue jeans, carrying a black bag. Motion: walking right.",
        4: "The person wears a black coat and gray pants. Motion: walking away from camera.",
    },
    "vehicle": {
        10: "A red sedan parked on the street. Motion: stopped.",
        11: "A white Budget moving truck with a yellow logo on the side. Motion: stopped.",
        12: "A white pickup truck turning right at the intersection. Motion: moving right.",
    },
    "incident": {100: "A man walks on the sidewalk."},
}
CP = C.Corpus(DOCS)


def plan(q):
    return P.rules_plan(q, cameras=["bus.G505", "school.G419"], vocab=CP.vocab)


def by_type(p, t):
    return [c for c in p["constraints"] if c["type"] == t]


def test_unknown_word_becomes_required_text_with_its_color():
    p = plan("red bluewhale")
    assert p["kind"] == "any"
    assert by_type(p, "text") == [P._c("text", "bluewhale")]
    assert by_type(p, "color") == [P._c("color", "red", "bluewhale")]
    assert [c["value"] for c in C.unsupported(p, CP)] == ["bluewhale"]


def test_red_bluewhale_never_counts_red_things_as_full_matches():
    p = plan("red bluewhale")
    hits = [dict(id=i, kind=k, score=1.0 / (1 + n)) for n, (k, i) in
            enumerate([("vehicle", 10), ("person", 1), ("person", 3), ("person", 4)])]
    out = C.score(p, hits, CP)
    assert all(h["match"] == "partial" for h in out)
    assert all('"bluewhale"' in h["missing"] for h in out)
    assert out[-1]["id"] == 4                      # the only non-red one ranks last among partials


def test_person_query_structure():
    p = plan("man in a red jacket with a backpack walking left")
    assert p["kind"] == "person"
    assert P._c("color", "red", "jacket") in p["constraints"]
    assert P._c("object", "jacket") in p["constraints"] and P._c("object", "backpack") in p["constraints"]
    assert P._c("action", "walking left") in p["constraints"]


def test_color_must_sit_on_the_right_garment():
    p = plan("person in red pants")
    hits = C.score(p, [dict(id=1, kind="person", score=1.0), dict(id=3, kind="person", score=0.5)], CP)
    assert all(h["match"] == "partial" for h in hits)  # red is on the hoodie/jacket, not the pants
    p = plan("red jacket")
    hits = C.score(p, [dict(id=1, kind="person", score=1.0), dict(id=3, kind="person", score=0.5)], CP)
    assert hits[0]["id"] == 3 and hits[0]["match"] == "full"


def test_vehicle_camera_and_turn():
    p = plan("white truck turning right on bus.G505")
    assert p["kind"] == "vehicle" and p["camera"] == "bus.G505"
    assert P._c("action", "turning right") in p["constraints"] and P._c("color", "white", "vehicle") in p["constraints"]
    hits = C.score(p, [dict(id=11, kind="vehicle", score=1.0), dict(id=12, kind="vehicle", score=0.2)], CP)
    assert [h["id"] for h in hits if h["match"] == "full"] == [12]


def test_brand_present_in_captions_is_found_and_full():
    p = plan("budget truck")
    assert C.unsupported(p, CP) == [] and P._c("object", "truck") in p["constraints"]
    assert CP.ids_with("vehicle", "budget") == [11]
    h = C.score(p, [dict(id=11, kind="vehicle", score=1.0)], CP)[0]
    assert h["match"] == "full"


def test_quoted_text_and_space_insensitive_match():
    p = plan('shirt with "whitman football"')
    assert P._c("text", "whitman football") in p["constraints"]
    assert CP.ids_with("person", "whitmanfootball") == [2]


def test_negation_excludes():
    p = plan("person without a backpack")
    assert P._c("object", "backpack") in p["negations"] and not by_type(p, "object")
    hits = C.score(p, [dict(id=1, kind="person", score=1.0), dict(id=4, kind="person", score=0.1)], CP)
    assert hits[0]["id"] == 4 and hits[-1]["match"] == "excluded"


def test_typo_is_corrected_from_caption_vocabulary():
    p = plan("red jakcet")
    assert P._c("object", "jacket") in p["constraints"] and any("jacket" in n for n in p["notes"])


def test_incident_kind():
    assert plan("two people fighting near the bus")["kind"] == "incident"


def test_finalize_cleans_llm_output():
    p = P.finalize({"kind": "weird", "constraints": [{"type": "color", "value": "Grey", "applies_to": "Vehicle"},
                                                    {"type": "bogus", "value": "x"}, {"type": "text", "value": " "}]})
    assert p["kind"] == "any" and p["constraints"] == [P._c("color", "gray", "vehicle")]


def test_real_words_are_not_spell_corrected_into_other_words():
    p = plan("blue whale")
    assert P._c("text", "whale") in p["constraints"] and P._c("color", "blue", "whale") in p["constraints"]
    p = plan("persn in a blu hoodie")
    assert p["kind"] == "person" and P._c("color", "blue", "hoodie") in p["constraints"]


def test_negated_words_leave_the_embedding_text():
    p = plan("woman in a black coat without a bag")
    assert "bag" not in p["text"] and "black coat" in p["text"]


def test_color_skips_a_modifier_to_reach_the_noun():
    p = plan("yellow school bus")
    assert P._c("color", "yellow", "vehicle") in p["constraints"]


def test_rare_caption_words_are_required_common_ones_soft():
    p = plan("guy in a whitman football shirt")
    terms = {c["value"]: c["required"] for c in by_type(p, "term")}
    assert terms == {"whitman football": True} or terms == {"whitman": True, "football": True}
    vocab = dict(CP.vocab, written=500)
    p = P.rules_plan('bag with "catalyst" written on it', vocab=vocab)
    assert {c["value"]: c["required"] for c in by_type(p, "term")} == {"written": False}


def test_caption_filter_finds_full_matches_and_respects_negation():
    p = plan("person without a backpack")
    assert 1 not in CP.satisfying("person", plan("person in a red hoodie without a backpack"))
    assert CP.satisfying("person", plan("red jacket")) == [3]


def test_color_stays_on_the_named_garment():
    cp = C.Corpus({"person": {7: "The person wears a red shirt under a black jacket and carries a backpack.",
                              8: "The person wears a red puffer jacket."}})
    p = P.rules_plan("red jacket", vocab=cp.vocab)
    hits = C.score(p, [dict(id=7, kind="person", score=1.0, upper_color="red"), dict(id=8, kind="person", score=0.1)], cp)
    assert [h["id"] for h in hits if h["match"] == "full"] == [8]


def test_merge_restores_dropped_requirements_and_fixes_typos():
    rules = plan("white truck turning right")
    llm = P.finalize({"kind": "vehicle", "text": "white truck", "planner": "llm",
                      "constraints": [{"type": "color", "value": "white", "applies_to": "vehicle"}]})
    m = P.merge(llm, rules)
    assert P._c("action", "turning right") in m["constraints"] and P._c("object", "truck") in m["constraints"]
    llm = P.finalize({"kind": "person", "constraints": [{"type": "color", "value": "blu hoodie"}]})
    m = P.merge(llm, plan("persn in a blu hoodie"))
    assert P._c("color", "blue", "hoodie") in m["constraints"]


def test_vlm_is_not_asked_about_motion_or_text_the_caption_already_reads():
    import pytest
    pytest.importorskip("torch")
    from src.query import verify as V
    h = {"caption": 'carries a blue bag with "Catalyst" written on it'}
    assert not V.checkable(P._c("action", "turning right"), "vehicle", h)
    assert not V.checkable(P._c("text", "catalyst"), "person", h)
    assert V.checkable(P._c("color", "blue", "bag"), "person", h)


def test_time_filters():
    assert P.parse_time("red car after 5pm")[0] == {"start": "17:00"}
    assert P.parse_time("between 4 and 5 pm on march 7")[0] == {"start": "16:00", "end": "17:00", "date": "*-03-07"}
    assert P.parse_time("around 5:15pm")[0] == {"start": "16:45", "end": "17:45"}
    assert P.parse_time("in the evening")[0] == {"start": "17:00", "end": "21:00"}
    assert P.parse_time("before 09:30 on 2018-03-07")[0] == {"end": "09:30", "date": "2018-03-07"}
    p = plan("red car after 5pm")
    assert p["time"] == {"start": "17:00"} and "5pm" not in p["text"] and p["kind"] == "vehicle"


def test_concepts_and_attributes():
    p = plan("cop")
    assert P._c("object", "uniform") in p["constraints"] and p["kind"] == "person"   # only implied -> required
    p = plan("old woman with a suitcase")
    assert P._c("attribute", "elderly", required=False) in p["constraints"]
    assert P._c("attribute", "woman", required=False) in p["constraints"]
    p = plan("cyclist in a yellow vest")
    assert P._c("object", "bicycle") in p["constraints"] and P._c("color", "yellow", "vest") in p["constraints"]
    assert C.evidence(P._c("attribute", "woman"), "person", "A woman wearing a black coat.", {}) == 1.0


def test_rewrite_matches_caption_style():
    assert P.rewrite(plan("man in a red jacket with a backpack")) == "A man wearing a red jacket, carrying a backpack."
    assert P.rewrite(plan("white truck turning right")).startswith("A white truck")
    assert P.rewrite(plan("red bluewhale")) == "red bluewhale"
