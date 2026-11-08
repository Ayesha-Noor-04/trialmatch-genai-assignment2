# TrialMatch — Criterion-Level Clinical Trial Eligibility with a Fine-Tuned Transformer

Educational screening prototype for the Generative AI course (Assignment 2).

**Not a clinical enrollment decision tool.** Predictions require human review.

Dataset: `ncbi/TrialGPT-Criterion-Annotations` only.  
Split: patient-grouped 37/8/8, seed 42.  
Base model: `Qwen/Qwen2.5-1.5B-Instruct`.

## Repository

GitHub: https://github.com/Ayesha-Noor-04/trialmatch-genai-assignment2

## Model Weights

The fine-tuned LoRA adapter is available separately on Hugging Face:

**Download:** https://huggingface.co/ayeshanr404/trialmatch-qwen2.5-1.5b-lora

The adapter is also included in this repository under:

`outputs/qwen15b_lora/best_adapter`

If the adapter folder is missing, download it using:

```bash
hf download ayeshanr404/trialmatch-qwen2.5-1.5b-lora \
  --local-dir outputs/qwen15b_lora/best_adapter
```

The base Qwen model is downloaded automatically from Hugging Face on first inference and cached locally.

## Running the Browser Application (Task 6)

### Requirements

- Python 3.10+
- Dependencies from `requirements.txt`
- Internet connection for the initial model download
- Sufficient memory to load the base model and adapter

### Setup

```bash
git clone https://github.com/Ayesha-Noor-04/trialmatch-genai-assignment2.git
cd trialmatch-genai-assignment2

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m pytest -q
```

### Launch

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

The application supports criterion-level eligibility screening, evidence sentence extraction, model comparison, benchmark visualization, and PDF/CSV report export.

## Task 1 Quick Start (Colab / Kaggle, Free T4)

```bash
pip install -r requirements.txt
python -m pytest -q
python -m trialmatch.split
python -m trialmatch.audit --max-len 2048
python -m trialmatch.evaluate --split val --tag base_val
python -m trialmatch.train_lora --out outputs/qwen15b_lora
python -m trialmatch.plot_curves --run outputs/qwen15b_lora
python -m trialmatch.evaluate --split val --adapter outputs/qwen15b_lora/best_adapter --tag lora_val

# Only once, after everything is frozen:
python -m trialmatch.audit --max-len 2048 --include-test
python -m trialmatch.evaluate --split test --allow-test --adapter outputs/qwen15b_lora/best_adapter --tag lora_test
python -m trialmatch.evaluate --split test --allow-test --tag base_test
```

Outputs include:

- `data/split_manifest.json`
- `outputs/data_audit.json`
- `outputs/<run>/train_log.jsonl`
- `outputs/<run>/training_curves.png`
- `outputs/eval/*_metrics.json`
- `outputs/eval/*_predictions.jsonl`

## Project Layout

| Path | Purpose |
|---|---|
| `trialmatch/data.py` | Dataset loading, cleaning, GPT-4 column removal, sentence splitting |
| `trialmatch/split.py` | Seed-42 patient split, manifest, leakage checks |
| `trialmatch/prompt.py` | Prompt, output schema, parser, tokenization |
| `trialmatch/train_lora.py` | LoRA supervised fine-tuning with completion-only loss |
| `trialmatch/evaluate.py` | Greedy evaluation, metrics, prediction dumps |
| `trialmatch/metrics.py` | Accuracy, macro-F1, per-class metrics, confusion matrix, evidence metrics |
| `trialmatch/kv_cache.py` | Explicit KV-cache decoding |
| `trialmatch/attention.py` | Reference and tiled online attention |
| `trialmatch/speculative.py` | Speculative decoding implementation |
| `app.py` | Streamlit browser application |
| `docs/task1_research.md` | Research sources, candidate models, design decisions |

## Tests

Run:

```bash
python -m pytest -q
```

All 14 tests currently pass.

## AI-Use Acknowledgement

Code and research notes were drafted with AI assistance under the student's supervision. All submitted code, experiments, and results were reviewed and tested by the student, who is responsible for understanding and explaining the implementation.