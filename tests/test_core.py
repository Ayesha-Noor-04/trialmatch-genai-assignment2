import numpy as np
import pandas as pd
import pytest

from trialmatch.data import clean_frame, parse_ids, split_note
from trialmatch.metrics import classification_metrics, evidence_metrics, evidence_report, summarize
from trialmatch.prompt import parse_output, target_json
from trialmatch.split import apply_split, check_no_leakage, make_split


# ---------- split ----------
def _frame(n_patients=53, rows_per=3):
    recs = []
    for p in range(n_patients):
        for k in range(rows_per):
            recs.append({"patient_id": f"sigir-{p}", "annotation_id": len(recs), "criterion_type": "inclusion",
                         "label": "included", "evidence_ids": [0]})
    return pd.DataFrame(recs)


def test_split_sizes_and_determinism():
    df = _frame()
    s1, s2 = make_split(df.patient_id), make_split(df.patient_id[::-1])
    assert s1 == s2  # independent of input order (sorted first)
    assert [len(s1[k]) for k in ("train", "val", "test")] == [37, 8, 8]


def test_split_matches_reference_recipe():
    df = _frame()
    ids = np.array(sorted(df.patient_id.unique()))
    np.random.default_rng(42).shuffle(ids)
    s = make_split(df.patient_id)
    assert s["train"] == ids[:37].tolist() and s["val"] == ids[37:45].tolist() and s["test"] == ids[45:].tolist()


def test_no_leakage_and_detects_overlap():
    df = _frame()
    s = make_split(df.patient_id)
    parts = apply_split(df, s)
    assert sum(len(v) for v in parts.values()) == len(df)
    bad = {**s, "val": s["val"] + [s["train"][0]]}
    with pytest.raises(AssertionError):
        check_no_leakage(df, bad)


def test_wrong_patient_count_rejected():
    with pytest.raises(ValueError):
        make_split([f"p{i}" for i in range(10)])


# ---------- data ----------
def test_split_note_ids_align():
    note = "0. A 58-year-old woman presents. 1. Pain began 2 days ago. 2. BP 150/90. 3. Denies smoking."
    s = split_note(note)
    assert len(s) == 4 and s[1] == "Pain began 2 days ago." and s[3] == "Denies smoking."


def test_split_note_number_inside_sentence():
    note = "0. Takes 2. 5 mg daily. 1. Second sentence. 2. Third."
    assert len(split_note(note)) == 3  # '2. 5 mg' is not mistaken for a marker before 1.


def test_parse_ids():
    assert parse_ids("[3, 1, 1]") == [1, 3] and parse_ids("[]") == [] and parse_ids(None) == []


def test_clean_frame_drops_gpt4_and_validates():
    raw = pd.DataFrame([{
        "annotation_id": 0, "patient_id": "p", "note": "0. x.", "trial_id": "T", "trial_title": "t",
        "criterion_type": "Inclusion", "criterion_text": "age", "gpt4_explanation": "LEAK",
        "explanation_correctness": "Correct", "gpt4_sentences": "[0]", "expert_sentences": "[0]",
        "gpt4_eligibility": "included", "expert_eligibility": "Included", "training": True}])
    df = clean_frame(raw)
    assert not any(c.startswith("gpt4") or c in ("explanation_correctness", "training") for c in df.columns)
    assert df.loc[0, "label"] == "included" and df.loc[0, "evidence_ids"] == [0]
    raw.loc[0, "expert_eligibility"] = "excluded"  # exclusion label on an inclusion criterion
    with pytest.raises(ValueError):
        clean_frame(raw)


# ---------- prompt/parse ----------
def test_parse_valid_json():
    p = parse_output(target_json([0, 2], "not included"), "inclusion", 5)
    assert p["label"] == "not included" and p["evidence_ids"] == [0, 2] and p["valid_json"]


def test_parse_rejects_wrong_label_set_and_bad_ids():
    p = parse_output('{"evidence": [1, 99], "label": "excluded"}', "inclusion", 3)
    assert p["label"] is None and p["evidence_ids"] == [1] and p["n_invalid_ids"] == 1


def test_parse_fallback_prefers_longest_label():
    p = parse_output('label: not included because ... "evidence": [1]', "inclusion", 3)
    assert p["label"] == "not included" and p["evidence_ids"] == [1] and not p["valid_json"]


# ---------- metrics ----------
def test_classification_known_values():
    yt = ["included", "included", "not included", "not applicable"]
    yp = ["included", "not included", "not included", "invalid"]
    m = classification_metrics(yt, yp)
    assert m["accuracy"] == 0.5
    assert m["per_class"]["included"]["recall"] == 0.5
    assert "invalid" in m["confusion_labels"] and "invalid" not in m["macro_labels"]


def test_evidence_empty_handling():
    # one correct abstention, one hit, one miss
    m = evidence_metrics([[], [1, 2], [3]], [[], [1], []])
    assert m["macro"]["f1"] == pytest.approx((1.0 + 2 / 3 + 0.0) / 3)
    assert m["micro"]["precision"] == 1.0 and m["micro"]["recall"] == pytest.approx(1 / 3)
    rep = evidence_report([[], [1, 2], [3]], [[], [1], []])
    assert rep["empty_gold"]["pred_also_empty_rate"] == 1.0
    assert rep["gold_nonempty_only"]["n"] == 2


def test_summarize_by_type():
    rows = [
        dict(criterion_type="inclusion", label="included", pred_label="included", gold_ids=[0], pred_ids=[0], valid_json=True),
        dict(criterion_type="exclusion", label="not excluded", pred_label=None, gold_ids=[], pred_ids=[], valid_json=False),
    ]
    s = summarize(rows)
    assert s["overall"]["classification"]["accuracy"] == 0.5
    assert s["inclusion"]["classification"]["accuracy"] == 1.0
    assert s["exclusion"]["classification"]["accuracy"] == 0.0
    assert s["label_valid_rate"] == 0.5
