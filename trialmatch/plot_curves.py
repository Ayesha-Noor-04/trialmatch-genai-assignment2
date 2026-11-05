"""Plot training/validation loss curves from train_log.jsonl.

  python -m trialmatch.plot_curves --run outputs/qwen15b_lora
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(Path(args.run) / "train_log.jsonl")]
    tr = [r for r in recs if r["type"] == "train"]
    va = [r for r in recs if r["type"] == "val"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([r["step"] for r in tr], [r["loss"] for r in tr], label="train loss (running mean)")
    ax.plot([r["step"] for r in va], [r["val_loss"] for r in va], "o-", label="validation loss (per epoch)")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("token-level cross-entropy")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out = Path(args.run) / "training_curves.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
