# Task 1 — Research notes, model selection and design decisions

> Status: DRAFT for the report (Sections: Related Work, Materials and Methods).
> Items marked **[VERIFY]** come from memory of model cards/papers and must be checked against the
> primary source before they go into the paper. Items marked **[CONFIRMED]** were read from the source
> during this project.

## 1. Sources (need >= 5; 10 listed)

| # | Source | Used for | Status |
|---|--------|----------|--------|
| 1 | Jin et al., "Matching patients to clinical trials with large language models", Nature Communications 15 (2024), arXiv:2307.15051 | Task definition, label sets, GPT-4 baseline (87.3% criterion accuracy), limitations (closed model, free-text only) | CONFIRMED |
| 2 | HF dataset card `ncbi/TrialGPT-Criterion-Annotations` | Schema (1,015 rows, 53 patients, 103 trials), columns, public-domain licence | CONFIRMED |
| 3 | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685 | Fine-tuning method | VERIFY |
| 4 | Dettmers et al., "QLoRA", arXiv:2305.14314 | Rejected alternative (4-bit) — not needed at 1.5B | VERIFY |
| 5 | Qwen Team, "Qwen2.5 Technical Report", arXiv:2412.15115 | Target/draft model family, shared tokenizer | VERIFY |
| 6 | Ainslie et al., "GQA", arXiv:2305.13245 | Explains small KV cache (Tasks 3, 5) | VERIFY |
| 7 | Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", arXiv:2211.17192 | Task 5 | VERIFY |
| 8 | Milakov & Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867 | Task 4 | VERIFY |
| 9 | Dao et al., "FlashAttention", arXiv:2205.14135 | Task 4 (tiling, IO-awareness) | VERIFY |
| 10 | Romanov & Shivade, "Lessons from NLI in the Clinical Domain" (EMNLP 2018) — MedNLI | Encoder/NLI baselines used in source 1 | CONFIRMED (cited in source 1) |

## 2. Model candidates (decoder-only, open weights)

Hardware assumption: single free Kaggle/Colab T4 (16 GB, fp16, **no bf16**).

| Candidate | Params | Context | Licence | Notes | Status |
|-----------|--------|---------|---------|-------|--------|
| **Qwen2.5-1.5B-Instruct** (target) | ~1.54B (1.31B non-embedding), 28 layers, GQA 12 Q / 2 KV heads | 32K | Apache-2.0 | Has a smaller same-tokenizer sibling for speculative decoding | VERIFY |
| **Qwen2.5-0.5B-Instruct** (draft) | ~0.49B, 24 layers, GQA 14 Q / 2 KV heads | 32K | Apache-2.0 | Identical tokenizer/vocab as target -> valid draft | VERIFY |
| Llama-3.2-1B-Instruct | ~1.24B | 128K | Llama 3.2 Community Licence (gated) | Gated download; licence restrictions; 3B sibling shares tokenizer | VERIFY |
| Qwen2.5-3B-Instruct | ~3.09B | 32K | Qwen Research Licence (non-commercial) | Licence less permissive; slower LoRA on T4 | VERIFY |
| Gemma-2-2B-it | ~2.6B | 8K | Gemma terms (gated) | Gated; 8K ctx fine but no same-family tiny draft | VERIFY |

**Decision (provisional):** target = Qwen2.5-1.5B-Instruct, draft = Qwen2.5-0.5B-Instruct.
Reasons: permissive licence, no gating, fp16 weights ~3 GB leaves ample T4 headroom for LoRA +
activations, GQA keeps KV cache tiny (useful for Tasks 3/5 measurements), and a verified-compatible
draft exists in the same family. To make the choice evidence-based rather than asserted, run the
zero-shot baseline for at least Qwen2.5-1.5B and one alternative (e.g. Llama-3.2-1B) on the
**validation** split and report both (A3 "alternatives considered").

Memory estimates to confirm from `config.json` / logged peak memory **[VERIFY]**:
* weights fp16: ~3.1 GB (1.5B), ~1.0 GB (0.5B)
* KV cache / token = 2 (K,V) x layers x kv_heads x head_dim x 2 bytes
  * 1.5B: 2 x 28 x 2 x 128 x 2 = 28,672 B (~28 KiB/token)
  * 0.5B: 2 x 24 x 2 x 64 x 2 = 12,288 B (~12 KiB/token)
* LoRA r=16 on all 7 linear projections of the 1.5B model: ~18.5M trainable params (~1.2%) —
  `train_lora.py` prints the exact number; use that.

## 3. Design decisions and rationale

* **Output schema**: `{"evidence": [ids], "label": "..."}`, evidence before label so the label is
  conditioned on cited sentences. Tradeoff: an ablation (label-first) is a cheap extra experiment.
* **No GPT-4 text anywhere.** The assignment forbids GPT-4 predictions as inputs and says they must not
  leak into evaluation. We also do **not** train on `gpt4_explanation`: it is a model output, not an
  expert label. Consequence: the model emits no free-text rationale. The app's "explanation" (Task 2/6)
  will be a deterministic template built only from the label and the cited sentences (never from
  absent evidence), which keeps it faithful by construction. State this limitation in the report.
* **Do not use the dataset's `training` column.** It is TrialGPT's own flag; the assignment prescribes a
  patient-grouped 37/8/8 split (seed 42).
* **Truncation:** none allowed. `encode_example` raises if a prompt exceeds `--max-len`; `audit.py`
  reports max token lengths per split so the "no truncation" claim is measured, not assumed.
* **Loss:** completion-only cross-entropy (prompt tokens masked), so the model is not trained to
  reproduce the note.
* **Model selection signal:** best adapter by validation loss; final reporting on validation macro-F1
  and (once) the test split. Thresholds/prompts are never tuned on test.
* **Known data property to discuss:** many "not excluded" rows have empty evidence (the annotator
  inferred it from absence), whereas "not enough information" is a different label. The assignment
  says absence must never automatically be proof — the report should show how often the model
  confuses these two classes (paper source 1 reports the same confusion for exclusion criteria).
* **Imbalance:** "not excluded" likely dominates; macro-F1 and per-class metrics are therefore the
  primary metrics, not accuracy. Class-weighting is a candidate follow-up if validation shows collapse.
