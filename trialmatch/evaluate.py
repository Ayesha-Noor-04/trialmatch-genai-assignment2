"""Greedy evaluation of a base model or base+LoRA adapter on the validation or test split.

  # before fine-tuning (same prompt, same decoding):
  python -m trialmatch.evaluate --split val --tag base_val
  # after fine-tuning:
  python -m trialmatch.evaluate --split val --adapter outputs/qwen15b_lora/best_adapter --tag lora_val
The test split is refused unless --allow-test is passed (final evaluation only).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from trialmatch.data import load_clean, split_note
from trialmatch.metrics import summarize
from trialmatch.prompt import build_messages, parse_output
from trialmatch.split import apply_split, load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--manifest", default="data/split_manifest.json")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="outputs/eval")
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to touch the test split without --allow-test (final evaluation only).")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device).eval()

    part = apply_split(load_clean(), load_split(args.manifest))[args.split]
    rows = part.to_dict("records")
    prompts = [tok.apply_chat_template(build_messages(r["note"], r["criterion_type"], r["criterion_text"]),
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    texts = [None] * len(rows)
    t0 = time.time()
    for s in range(0, len(order), args.batch_size):
        idx = order[s : s + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 repetition_penalty=1.0,  # Qwen's generation_config defaults to 1.05
                                 pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, d in zip(idx, dec):
            texts[i] = d
    elapsed = time.time() - t0

    preds, eval_rows = [], []
    for r, text in zip(rows, texts):
        p = parse_output(text, r["criterion_type"], len(split_note(r["note"])))
        preds.append({"annotation_id": int(r["annotation_id"]), "patient_id": r["patient_id"],
                      "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"],
                      "gold_label": r["label"], "gold_ids": r["evidence_ids"],
                      "pred_label": p["label"], "pred_ids": p["evidence_ids"],
                      "valid_json": p["valid_json"], "n_invalid_ids": p["n_invalid_ids"], "raw_output": text})
        eval_rows.append({"criterion_type": r["criterion_type"], "label": r["label"], "pred_label": p["label"],
                          "gold_ids": r["evidence_ids"], "pred_ids": p["evidence_ids"], "valid_json": p["valid_json"]})

    metrics = summarize(eval_rows)
    metrics["run"] = {"tag": args.tag, "model": args.model, "adapter": args.adapter, "split": args.split,
                      "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                                   "repetition_penalty": 1.0, "batch_size": args.batch_size, "dtype": str(dtype)},
                      "wall_time_s": elapsed, "torch": torch.__version__}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out_dir / f"{args.tag}_predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    c = metrics["overall"]["classification"]
    print(f"[{args.tag}] n={metrics['n']} acc={c['accuracy']:.3f} macroF1={c['macro_f1']:.3f} "
          f"json_valid={metrics['json_valid_rate']:.3f} label_valid={metrics['label_valid_rate']:.3f}")


if __name__ == "__main__":
    main()
