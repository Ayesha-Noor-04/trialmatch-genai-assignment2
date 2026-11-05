"""Data audit: label/ID sanity, token lengths per split, and a truncation check.

Run:  python -m trialmatch.audit --model Qwen/Qwen2.5-1.5B-Instruct --max-len 2048
Writes outputs/data_audit.json. Fails loudly if any example would need truncation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trialmatch.data import load_clean, split_note
from trialmatch.prompt import encode_example
from trialmatch.split import apply_split, load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--manifest", default="data/split_manifest.json")
    ap.add_argument("--out", default="outputs/data_audit.json")
    ap.add_argument(
        "--include-test", action="store_true",
        help="Only use after all model/prompt choices are frozen, for final test audit."
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    df = load_clean()
    parts = apply_split(df, load_split(args.manifest))

    report = {"model_tokenizer": args.model, "max_len": args.max_len, "splits": {}}
    problems = []
    split_names = ["train", "val"]
    if args.include_test:
        split_names.append("test")
    for name in split_names:
        part = parts[name]
        lens, bad_ids, n_sents = [], 0, []
        for _, row in part.iterrows():
            ns = len(split_note(row["note"]))
            n_sents.append(ns)
            if any(i >= ns for i in row["evidence_ids"]):
                bad_ids += 1
            try:
                lens.append(encode_example(tok, row, 10**9)["total_len"])
            except Exception as e:  # pragma: no cover
                problems.append(str(e))
        lens = np.array(lens)
        report["splits"][name] = {
            "rows": int(len(part)),
            "patients": int(part["patient_id"].nunique()),
            "tokens_mean": float(lens.mean()),
            "tokens_p95": float(np.percentile(lens, 95)),
            "tokens_max": int(lens.max()),
            "rows_over_max_len": int((lens > args.max_len).sum()),
            "rows_with_evidence_id_out_of_range": bad_ids,
            "mean_sentences_per_note": float(np.mean(n_sents)),
            "rows_with_empty_gold_evidence": int((part["evidence_ids"].map(len) == 0).sum()),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    over = sum(s["rows_over_max_len"] for s in report["splits"].values())
    assert over == 0, f"{over} rows exceed max_len; raise --max-len (no silent truncation allowed)"
    assert not problems, problems[:3]


if __name__ == "__main__":
    main()
