# TrialMatch report figures

These figures were generated from the saved TrialMatch experiment outputs and the benchmark values recorded in the project logs. They are intended for the IEEE report and repository.

## Files
- `fig1_dataset_split.png`: patient-grouped train/validation/test split.
- `fig2_classification_performance.png`: base and LoRA accuracy/macro-F1 on validation and test sets.
- `fig3_lora_test_confusion_matrix.png`: fine-tuned model test confusion matrix.
- `fig4_evidence_f1.png`: evidence extraction F1, including gold-nonempty-only scoring.
- `fig5_kv_cache_latency.png`: cached vs. uncached decoding latency.
- `fig6_attention_latency.png`: attention implementation latency.
- `fig7_attention_memory.png`: attention implementation peak extra memory.
- `fig8_speculative_latency.png`: ordinary vs. speculative decoding latency.
- `fig9_speculative_acceptance.png`: draft-token acceptance rate.

## Notes for interpretation
- The test set contains no gold `excluded` examples; the confusion matrix therefore uses the five labels present in the LoRA test metrics.
- Speculative decoding was slower than ordinary decoding in these measurements (mean speedup 0.553x). Do not describe it as a speedup.
- The tiled online attention implementation reduced temporary memory relative to the reference implementation but was slower in this Python implementation. PyTorch SDPA was fastest in these measurements.
- The KV-cache benchmark timings are hardware/runtime measurements and may vary across runs.
- `training_curves.png` already exists under `outputs/qwen15b_lora/` in the project; copy it into this folder if the report needs a training-curve figure.
