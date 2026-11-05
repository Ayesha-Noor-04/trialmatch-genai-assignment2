# TrialMatch — criterion-level clinical-trial eligibility with a fine-tuned Transformer

Educational screening prototype for the Generative AI course (Assignment 2). **Not a clinical
enrollment decision tool.**

Dataset: `ncbi/TrialGPT-Criterion-Annotations` (only). Split: patient-grouped 37/8/8, seed 42.

## Task 1 quick start (Kaggle / Colab, free T4)

```bash
git clone <your-repo-url> && cd trialmatch-repo
pip install -r requirements.txt
python -m pytest -q                              # 14 tests, no GPU needed
python -m trialmatch.split                       # writes data/split_manifest.json
python -m trialmatch.audit --max-len 2048        # audits train/val only; test stays untouched
python -m trialmatch.evaluate --split val --tag base_val                       # BEFORE fine-tuning
python -m trialmatch.train_lora --out outputs/qwen15b_lora                     # LoRA fine-tune
python -m trialmatch.plot_curves --run outputs/qwen15b_lora
python -m trialmatch.evaluate --split val --adapter outputs/qwen15b_lora/best_adapter --tag lora_val
# Only once, when everything is frozen:
python -m trialmatch.audit --max-len 2048 --include-test  # optional final test truncation audit
python -m trialmatch.evaluate --split test --allow-test --adapter outputs/qwen15b_lora/best_adapter --tag lora_test
python -m trialmatch.evaluate --split test --allow-test --tag base_test
```

Outputs: `data/split_manifest.json`, `outputs/data_audit.json`, `outputs/<run>/train_log.jsonl`,
`outputs/<run>/training_curves.png`, `outputs/eval/*_metrics.json`, `outputs/eval/*_predictions.jsonl`.

## Layout

| Path | Purpose |
|------|---------|
| `trialmatch/data.py` | load/clean dataset, drop GPT-4 columns, sentence splitting |
| `trialmatch/split.py` | seed-42 patient split, manifest, leakage checks |
| `trialmatch/prompt.py` | prompt, output schema, parser, tokenisation (no silent truncation) |
| `trialmatch/train_lora.py` | LoRA SFT with completion-only loss |
| `trialmatch/evaluate.py` | greedy eval, metrics, predictions dump (test needs `--allow-test`) |
| `trialmatch/metrics.py` | accuracy, macro-F1, per-class P/R/F1, confusion, evidence P/R/F1 |
| `docs/task1_research.md` | sources, candidate models, design decisions |

## AI-use acknowledgement
Code and research notes were drafted with Claude (Anthropic) under the student's supervision; all code is
tested and must be understood by the student before submission.
