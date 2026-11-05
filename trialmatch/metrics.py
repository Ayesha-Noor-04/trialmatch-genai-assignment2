"""Criterion classification and evidence-sentence metrics.

Conventions (documented for the report):
* Classification: accuracy over all rows; macro-F1 over labels that occur in gold or
  predictions (an unparseable prediction is class "invalid": it counts as an error via
  recall of the gold class but does not add an extra class to the macro average).
* Evidence: micro = pooled TP/FP/FN over records. macro = mean of per-record P/R/F1.
  Per-record rule: gold empty AND pred empty -> P=R=F1=1 (correct abstention);
  exactly one of them empty -> 0/0/0; otherwise standard set overlap.
  Reported for (a) all records and (b) only records with non-empty gold evidence.
  Absent evidence is never treated as proof of eligibility or ineligibility.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from trialmatch.data import LABELS

CANON = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]
INVALID = "invalid"


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    if not y_true:
        return {"n": 0}
    present = [l for l in CANON if l in set(y_true) | set(y_pred)]
    macro_labels = list(present)
    cm_labels = present + ([INVALID] if INVALID in set(y_pred) else [])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=cm_labels, zero_division=0)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    idx = [cm_labels.index(l) for l in macro_labels]
    return {
        "n": len(y_true),
        "accuracy": acc,
        "macro_f1": float(np.mean([f[i] for i in idx])),
        "macro_labels": macro_labels,
        "per_class": {
            l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, l in enumerate(cm_labels)
        },
        "confusion_labels": cm_labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=cm_labels).tolist(),
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evidence_metrics(gold: list[list[int]], pred: list[list[int]]) -> dict:
    if not gold:
        return {"n": 0}
    tp = fp = fn = 0
    per = []
    for g, q in zip(gold, pred):
        g, q = set(g), set(q)
        t, a, b = len(g & q), len(q - g), len(g - q)
        tp, fp, fn = tp + t, fp + a, fn + b
        if not g and not q:
            per.append((1.0, 1.0, 1.0))
        else:
            P = t / len(q) if q else 0.0
            R = t / len(g) if g else 0.0
            F = 2 * P * R / (P + R) if P + R else 0.0
            per.append((P, R, F))
    mp, mr, mf = _prf(tp, fp, fn)
    arr = np.array(per)
    return {
        "n": len(gold),
        "micro": {"precision": mp, "recall": mr, "f1": mf},
        "macro": {"precision": float(arr[:, 0].mean()), "recall": float(arr[:, 1].mean()), "f1": float(arr[:, 2].mean())},
    }


def evidence_report(gold: list[list[int]], pred: list[list[int]]) -> dict:
    ne = [i for i, g in enumerate(gold) if g]
    empty = [i for i, g in enumerate(gold) if not g]
    return {
        "all_records": evidence_metrics(gold, pred),
        "gold_nonempty_only": evidence_metrics([gold[i] for i in ne], [pred[i] for i in ne]),
        "empty_gold": {
            "n": len(empty),
            "pred_also_empty_rate": float(np.mean([not pred[i] for i in empty])) if empty else None,
        },
    }


def summarize(rows: list[dict]) -> dict:
    """rows: dicts with criterion_type, label, pred_label (None allowed), gold_ids, pred_ids, valid_json."""
    out = {"n": len(rows)}
    if not rows:
        return out
    out["json_valid_rate"] = float(np.mean([r["valid_json"] for r in rows]))
    out["label_valid_rate"] = float(np.mean([r["pred_label"] is not None for r in rows]))

    def sub(rs):
        yt = [r["label"] for r in rs]
        yp = [r["pred_label"] if r["pred_label"] is not None else INVALID for r in rs]
        return {
            "classification": classification_metrics(yt, yp),
            "evidence": evidence_report([r["gold_ids"] for r in rs], [r["pred_ids"] for r in rs]),
        }

    out["overall"] = sub(rows)
    for ctype in LABELS:
        out[ctype] = sub([r for r in rows if r["criterion_type"] == ctype])
    return out
