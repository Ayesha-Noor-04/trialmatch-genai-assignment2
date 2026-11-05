"""LoRA supervised fine-tuning (completion-only loss) on the fixed training split.

Example (Kaggle/Colab T4):
  python -m trialmatch.train_lora --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/qwen15b_lora
Selects the best adapter by validation loss; the test split is never loaded here.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from trialmatch.data import load_clean
from trialmatch.prompt import encode_example
from trialmatch.split import apply_split, load_split


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


@torch.no_grad()
def val_loss(model, loader, device, amp_dtype):
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(**b)
        ntok = int((b["labels"][:, 1:] != -100).sum())
        tot += out.loss.item() * ntok
        n += ntok
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--manifest", default="data/split_manifest.json")
    ap.add_argument("--out", default="outputs/qwen15b_lora")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype, amp_dtype = torch.bfloat16, torch.bfloat16
    elif device.type == "cuda":
        dtype, amp_dtype = torch.float16, torch.float16
    else:
        dtype, amp_dtype = torch.float32, None
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    df = load_clean()
    parts = apply_split(df, load_split(args.manifest))  # test part exists but is never encoded here
    train_ds = [encode_example(tok, r, args.max_len) for _, r in parts["train"].iterrows()]
    val_ds = [encode_example(tok, r, args.max_len) for _, r in parts["val"].iterrows()]
    print(f"train={len(train_ds)} val={len(val_ds)} max_tokens={max(e['total_len'] for e in train_ds + val_ds)}")

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                          collate_fn=lambda b: collate(b, tok.pad_token_id))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.to(device).train()

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)

    (out / "config.json").write_text(json.dumps({
        **vars(args), "dtype": str(dtype), "trainable_params": n_trainable,
        "train_rows": len(train_ds), "val_rows": len(val_ds), "total_optimizer_steps": total_steps,
        "torch": torch.__version__,
    }, indent=2))
    log = open(out / "train_log.jsonl", "w")

    best, bad, step, t0 = float("inf"), 0, 0, time.time()
    run_loss, run_n = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        for i, b in enumerate(train_dl, 1):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**b).loss
            scaler.scale(loss / args.grad_accum).backward()
            run_loss += loss.item(); run_n += 1
            if i % args.grad_accum == 0 or i == len(train_dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1
                if step % args.log_every == 0:
                    rec = {"type": "train", "step": step, "epoch": epoch, "loss": run_loss / run_n,
                           "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
                    log.write(json.dumps(rec) + "\n"); log.flush(); print(rec)
                    run_loss, run_n = 0.0, 0
        vl = val_loss(model, val_dl, device, amp_dtype)
        rec = {"type": "val", "step": step, "epoch": epoch, "val_loss": vl, "elapsed_s": time.time() - t0}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(rec)
        if vl < best:
            best, bad = vl, 0
            model.save_pretrained(out / "best_adapter")
            tok.save_pretrained(out / "best_adapter")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}")
                break
    print(f"best val loss {best:.4f}; adapter in {out/'best_adapter'}; peak GPU mem "
          f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB" if device.type == "cuda" else f"best val loss {best:.4f}")


if __name__ == "__main__":
    main()
