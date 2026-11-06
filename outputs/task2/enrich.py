
import json
from pathlib import Path
from .data import load_clean, split_note

def build_note_lookup():
    df = load_clean()
    return {
        int(row["annotation_id"]): row["note"]
        for _, row in df.iterrows()
    }

def enrich_prediction(prediction, notes):
    note = notes[int(prediction["annotation_id"])]
    sentences = split_note(note)

    ids = [
        i for i in prediction.get("pred_ids", [])
        if isinstance(i, int) and 0 <= i < len(sentences)
    ]

    evidence = [
        {"sentence_id": i, "text": sentences[i]}
        for i in ids
    ]

    label = prediction.get("pred_label")
    criterion_type = prediction["criterion_type"]

    if evidence:
        explanation = (
            f"Predicted category: {label}. "
            "The model cited these patient-note sentences as evidence: "
            + " ".join(
                f"[{item['sentence_id']}] {item['text']}"
                for item in evidence
            )
        )
    else:
        explanation = (
            f"Predicted category: {label}. "
            "The model did not cite a supporting sentence. "
            "Absence of cited evidence does not establish eligibility "
            "or ineligibility."
        )

    return {
        "annotation_id": prediction["annotation_id"],
        "criterion_type": criterion_type,
        "criterion": prediction["criterion_text"],
        "predicted_category": label,
        "cited_sentence_ids": ids,
        "evidence_text": evidence,
        "explanation": explanation,
        "gold_category": prediction["gold_label"],
        "gold_sentence_ids": prediction["gold_ids"],
        "raw_model_output": prediction["raw_output"]
    }

def enrich_file(source, destination, notes):
    with open(source) as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    enriched = [enrich_prediction(p, notes) for p in predictions]

    with open(destination, "w") as f:
        for item in enriched:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")

    return len(enriched)
