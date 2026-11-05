"""Prompt construction, output schema, output parsing and tokenisation for SFT.

Output schema (a single JSON object, evidence first so the label is conditioned on it):
    {"evidence": [<sentence ids>], "label": "<one label from the criterion-type label set>"}
No GPT-4 explanation/prediction text is used anywhere (assignment leakage rule).
"""
from __future__ import annotations

import json
import re

from trialmatch.data import LABELS, split_note

SYSTEM = (
    "You are a careful clinical-trial screening assistant. Use only the patient note. "
    "Respond with a single JSON object and nothing else."
)

DEFINITIONS = {
    "inclusion": {
        "included": "the note shows the patient meets this criterion",
        "not included": "the note shows the patient does not meet this criterion",
        "not enough information": "the note lacks the information needed to decide",
        "not applicable": "the criterion does not apply to this patient",
    },
    "exclusion": {
        "excluded": "the note shows the patient meets this exclusion criterion (so is excluded)",
        "not excluded": "the note shows the patient does not meet this exclusion criterion",
        "not enough information": "the note lacks the information needed to decide",
        "not applicable": "the criterion does not apply to this patient",
    },
}


def build_messages(note: str, criterion_type: str, criterion_text: str) -> list[dict]:
    sents = split_note(note)
    note_block = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
    defs = "\n".join(f'- "{k}": {v}' for k, v in DEFINITIONS[criterion_type].items())
    user = (
        f"Decide the patient's status for one clinical-trial {criterion_type} criterion.\n\n"
        f"Allowed labels:\n{defs}\n\n"
        f"Patient note (sentence ID in brackets):\n{note_block}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        'Answer as JSON: {"evidence": [sentence IDs that support the label, [] if none], '
        '"label": "<allowed label>"}'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def target_json(evidence_ids: list[int], label: str) -> str:
    return json.dumps({"evidence": list(evidence_ids), "label": label}, ensure_ascii=False)


def parse_output(text: str, criterion_type: str, n_sentences: int) -> dict:
    """Parse model output. Never raises. label is None if no valid label could be recovered."""
    allowed = LABELS[criterion_type]
    res = {"label": None, "evidence_ids": [], "valid_json": False, "n_invalid_ids": 0}

    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None

    raw_ids = []
    if isinstance(obj, dict):
        res["valid_json"] = True
        lab = obj.get("label")
        if isinstance(lab, str) and lab.strip().lower() in allowed:
            res["label"] = lab.strip().lower()
        ev = obj.get("evidence", [])
        raw_ids = ev if isinstance(ev, list) else []
    else:  # fallback: regex recovery so a malformed output is still inspectable
        for lab in sorted(allowed, key=len, reverse=True):
            if re.search(rf'(?<!\w){re.escape(lab)}(?!\w)', text.lower()):
                res["label"] = lab
                break
        em = re.search(r'"evidence"\s*:\s*\[([^\]]*)\]', text)
        if em:
            raw_ids = re.findall(r"\d+", em.group(1))

    ids = []
    for v in raw_ids:
        try:
            i = int(v)
        except (TypeError, ValueError):
            res["n_invalid_ids"] += 1
            continue
        if 0 <= i < n_sentences:
            ids.append(i)
        else:
            res["n_invalid_ids"] += 1
    res["evidence_ids"] = sorted(set(ids))
    return res


def encode_example(tok, row, max_len: int) -> dict:
    """Tokenise one example for completion-only SFT. Raises instead of silently truncating."""
    msgs = build_messages(row["note"], row["criterion_type"], row["criterion_text"])
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target_json(row["evidence_ids"], row["label"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
    n = len(p_ids) + len(t_ids)
    if n > max_len:
        raise ValueError(
            f"annotation_id={row['annotation_id']} needs {n} tokens > max_len={max_len}; "
            "refusing to truncate (could drop evidence). Raise --max-len."
        )
    return {
        "input_ids": p_ids + t_ids,
        "labels": [-100] * len(p_ids) + t_ids,
        "prompt_len": len(p_ids),
        "total_len": n,
    }
