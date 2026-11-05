"""Patient-grouped 37/8/8 split (seed 42) with manifest and leakage checks.

Protocol from the assignment: sort unique patient IDs, shuffle with NumPy
default_rng(42), first 37 -> train, next 8 -> validation, remaining 8 -> test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8


def make_split(patient_ids) -> dict[str, list[str]]:
    ids = np.array(sorted({str(p) for p in patient_ids}))
    if len(ids) != N_TRAIN + N_VAL + N_TEST:
        raise ValueError(
            f"Expected {N_TRAIN + N_VAL + N_TEST} unique patients, found {len(ids)}."
        )
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    return {
        "train": ids[:N_TRAIN].tolist(),
        "val": ids[N_TRAIN : N_TRAIN + N_VAL].tolist(),
        "test": ids[N_TRAIN + N_VAL :].tolist(),
    }


def check_no_leakage(df: pd.DataFrame, split: dict[str, list[str]]) -> None:
    sets = {k: set(v) for k, v in split.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            assert not overlap, f"Patient leakage between {a} and {b}: {sorted(overlap)}"
    assert set().union(*sets.values()) == set(df["patient_id"]), "Split does not cover all patients"
    owner = {p: k for k, v in split.items() for p in v}
    assert df["patient_id"].map(owner).notna().all(), "Row with unassigned patient"


def apply_split(df: pd.DataFrame, split: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    check_no_leakage(df, split)
    return {k: df[df["patient_id"].isin(v)].reset_index(drop=True) for k, v in split.items()}


def write_manifest(df: pd.DataFrame, split: dict[str, list[str]], path: str | Path) -> dict:
    parts = apply_split(df, split)
    manifest = {
        "dataset": "ncbi/TrialGPT-Criterion-Annotations",
        "seed": SEED,
        "method": "sorted unique patient_id -> np.random.default_rng(42).shuffle -> 37/8/8",
        "patients": {k: len(v) for k, v in split.items()},
        "rows": {k: int(len(parts[k])) for k in split},
        "patient_ids": split,
        "ids_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split(path: str | Path = "data/split_manifest.json") -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())["patient_ids"]


def main() -> None:
    from trialmatch.data import load_clean

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/split_manifest.json")
    args = ap.parse_args()
    df = load_clean()
    split = make_split(df["patient_id"].unique())
    m = write_manifest(df, split, args.out)
    print(json.dumps({k: m[k] for k in ("patients", "rows", "ids_sha256")}, indent=2))
    print(f"Manifest written to {args.out}")


if __name__ == "__main__":
    main()
