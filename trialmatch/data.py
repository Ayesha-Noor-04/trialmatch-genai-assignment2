"""Dataset loading, cleaning and patient-note sentence splitting.

Leakage policy (assignment requirement): only the columns in KEEP_COLS are ever used.
The `gpt4_*` columns, `explanation_correctness` and the dataset's own `training` flag are
dropped immediately on load and never reach prompts, training targets or evaluation.
"""
from __future__ import annotations

import ast
import re

import pandas as pd

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}

KEEP_COLS = [
    "annotation_id",
    "patient_id",
    "note",
    "trial_id",
    "trial_title",
    "criterion_type",
    "criterion_text",
    "expert_sentences",
    "expert_eligibility",
]


def parse_ids(value) -> list[int]:
    """Parse the dataset's sentence-id cell (e.g. '[0, 1, 2]') into a sorted list of ints."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return sorted({int(v) for v in value})
    if isinstance(value, float) and pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    return sorted({int(v) for v in ast.literal_eval(s)})


def clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only allowed columns, normalise labels, parse evidence ids, validate label sets."""
    missing = [c for c in KEEP_COLS if c not in raw.columns]
    if missing:
        raise KeyError(f"Dataset is missing expected columns: {missing}. Found: {list(raw.columns)}")

    df = raw[KEEP_COLS].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["criterion_type"] = df["criterion_type"].astype(str).str.strip().str.lower()
    df["expert_eligibility"] = df["expert_eligibility"].astype(str).str.strip().str.lower()

    n_before = len(df)
    df = df[df["criterion_text"].notna() & (df["criterion_text"].astype(str).str.strip() != "")]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[data] dropped {n_dropped} rows with empty criterion_text")

    bad_type = sorted(set(df["criterion_type"]) - set(LABELS))
    if bad_type:
        raise ValueError(f"Unexpected criterion_type values: {bad_type}")
    for ctype, allowed in LABELS.items():
        got = set(df.loc[df["criterion_type"] == ctype, "expert_eligibility"])
        unexpected = sorted(got - set(allowed))
        if unexpected:
            raise ValueError(f"Unexpected {ctype} labels {unexpected}; allowed {allowed}")

    df["evidence_ids"] = df["expert_sentences"].map(parse_ids)
    df = df.drop(columns=["expert_sentences"]).rename(columns={"expert_eligibility": "label"})
    return df.reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    from datasets import load_dataset  # lazy: keeps unit tests light

    raw = load_dataset(DATASET_ID, split="train").to_pandas()
    return clean_frame(raw)


def split_note(note: str) -> list[str]:
    """Split a note formatted as '0. First sentence. 1. Second sentence. ...' into sentences.

    The list index equals the sentence ID used by the dataset's annotations. Markers are
    located sequentially (0, 1, 2, ...), so numbers inside sentence text cannot create a
    false split unless they exactly mimic the next expected marker.
    """
    starts: list[tuple[int, int]] = []  # (marker_start, text_start)
    pos = 0
    idx = 0
    while True:
        m = re.compile(rf"(?<!\S){idx}\.\s").search(note, pos)
        if not m:
            break
        starts.append((m.start(), m.end()))
        pos = m.end()
        idx += 1
    if not starts:
        return [note.strip()]
    sents = []
    for i, (_, text_start) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(note)
        sents.append(note[text_start:end].strip())
    return sents
