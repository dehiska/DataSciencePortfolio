"""
Vertex AI custom training job — YouTube toxicity RoBERTa classifier.

Architecture: roberta-base → [CLS] pooling → Dropout(0.3) → Linear(768, 3)
Loss:         BCEWithLogitsLoss with class-weighted pos_weight
Uncertainty:  MC Dropout (keep dropout active at inference time)

Data contract (CSV columns required):
    text, label_toxicity, label_hate_racism, label_harassment, source

Usage (local):
    python trainer/train.py --data-uri data/processed/training_data.csv

Usage (Vertex AI — env vars set automatically):
    AIP_MODEL_DIR=gs://bucket/models/job-name/ python trainer/train.py \
        --data-uri gs://bucket/data/training_data.csv
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizerFast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

LABEL_COLS = ["label_toxicity", "label_hate_racism", "label_harassment"]


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train RoBERTa toxicity classifier")
    p.add_argument("--data-uri",    default="data/processed/training_data.csv",
                   help="Local path or gs:// URI to training CSV")
    p.add_argument("--model-name",  default="roberta-base")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--max-length",  type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--max-steps",   type=int,   default=-1,
                   help="Stop after this many optimizer steps (-1 = no limit)")
    p.add_argument("--sample-frac", type=float, default=1.0,
                   help="Fraction of training data to use (0.0-1.0, default=1.0 = all)")
    # Vertex AI sets AIP_MODEL_DIR to gs://bucket/models/<job>/ automatically
    p.add_argument("--output-dir",
                   default=os.environ.get("AIP_MODEL_DIR", "models/"))
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_csv(uri: str) -> pd.DataFrame:
    if uri.startswith("gs://"):
        local = "/tmp/training_data.csv"
        log.info("Downloading %s → %s via gsutil", uri, local)
        subprocess.run(["gsutil", "cp", uri, local], check=True)
        return pd.read_csv(local)
    return pd.read_csv(uri)


# ── Model ─────────────────────────────────────────────────────────────────────

class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts  = df["text"].tolist()
        self.labels = df[LABEL_COLS].values.astype("float32")
        self.tok    = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx]),
        }


class ToxicityClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.3):
        super().__init__()
        self.roberta    = RobertaModel.from_pretrained(model_name)
        hidden          = self.roberta.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state[:, 0, :]))


# ── Train / Eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, global_step=0, max_steps=-1):
    model.train()
    total = 0.0
    for i, batch in enumerate(loader):
        if max_steps > 0 and global_step >= max_steps:
            break
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(ids, mask), lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        global_step += 1
        if (i + 1) % 50 == 0:
            log.info("  batch %d/%d  step %d  loss=%.4f", i + 1, len(loader), global_step, loss.item())
    return total / max(i + 1, 1), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    logit_list, label_list, total = [], [], 0.0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        logits = model(ids, mask)
        total += criterion(logits, lbls).item()
        logit_list.append(torch.sigmoid(logits).cpu().numpy())
        label_list.append(lbls.cpu().numpy())

    probs  = np.vstack(logit_list)
    labels = np.vstack(label_list)
    preds  = (probs >= 0.5).astype(int)

    metrics = {"val_loss": total / len(loader)}
    for i, col in enumerate(LABEL_COLS):
        name = col.replace("label_", "")
        if labels[:, i].sum() > 0:
            metrics[f"f1_{name}"]    = f1_score(labels[:, i], preds[:, i], zero_division=0)
            metrics[f"prauc_{name}"] = average_precision_score(labels[:, i], probs[:, i])
        else:
            metrics[f"f1_{name}"] = metrics[f"prauc_{name}"] = 0.0

    return metrics


# ── Artifact saving ───────────────────────────────────────────────────────────

def save_artifacts(local_model: str, local_meta: str, local_tok: str, output_dir: str):
    if output_dir.startswith("gs://"):
        dest = output_dir.rstrip("/")
        subprocess.run(["gsutil", "cp", local_model, dest + "/"], check=True)
        subprocess.run(["gsutil", "cp", local_meta,  dest + "/"], check=True)
        subprocess.run(["gsutil", "-m", "cp", "-r", local_tok, dest + "/"], check=True)
        log.info("Artifacts uploaded to %s", dest)
    else:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_model, out / Path(local_model).name)
        shutil.copy(local_meta,  out / Path(local_meta).name)
        tok_dest = out / "tokenizer"
        if tok_dest.exists():
            shutil.rmtree(tok_dest)
        shutil.copytree(local_tok, tok_dest)
        log.info("Artifacts saved to %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=== Toxicity Training Job ===")
    log.info("Device:    %s", device)
    log.info("Data:      %s", args.data_uri)
    log.info("Output:    %s", args.output_dir)
    log.info("Epochs:    %d  Batch: %d  LR: %s  MaxLen: %d",
             args.epochs, args.batch_size, args.lr, args.max_length)

    # ── Data ──────────────────────────────────────────────────────
    log.info("Loading data...")
    df = load_csv(args.data_uri)
    log.info("Loaded %d rows", len(df))

    for col in LABEL_COLS:
        pos = int(df[col].sum())
        log.info("  %s: %d positive (%.1f%%)", col, pos, pos / len(df) * 100)

    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
        log.info("Sampled %.0f%% → %d rows", args.sample_frac * 100, len(df))

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=args.seed, stratify=df["label_toxicity"]
    )
    log.info("Train: %d  Val: %d", len(train_df), len(val_df))

    # ── Tokenizer ─────────────────────────────────────────────────
    log.info("Loading tokenizer: %s", args.model_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)

    n_workers = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(
        ToxicityDataset(train_df, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True, num_workers=n_workers,
    )
    val_loader = DataLoader(
        ToxicityDataset(val_df, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=False, num_workers=n_workers,
    )

    # ── Model ─────────────────────────────────────────────────────
    log.info("Loading model: %s", args.model_name)
    model = ToxicityClassifier(args.model_name, dropout=args.dropout).to(device)

    pos_counts  = train_df[LABEL_COLS].sum().values
    neg_counts  = len(train_df) - pos_counts
    pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)
    log.info("pos_weights: %s", pos_weights.tolist())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ── Training loop ─────────────────────────────────────────────
    best_f1        = 0.0
    global_step    = 0
    local_model_pt = "/tmp/roberta_toxicity_best.pt"
    local_meta     = "/tmp/model_meta.json"
    local_tok_dir  = "/tmp/tokenizer"

    if args.max_steps > 0:
        log.info("Max steps: %d (will stop early if reached)", args.max_steps)

    for epoch in range(1, args.epochs + 1):
        if args.max_steps > 0 and global_step >= args.max_steps:
            log.info("Reached max_steps=%d — stopping training.", args.max_steps)
            break
        log.info("--- Epoch %d/%d ---", epoch, args.epochs)
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, global_step, args.max_steps
        )
        metrics    = evaluate(model, val_loader, criterion, device)
        avg_f1     = np.mean([metrics[f"f1_{c.replace('label_', '')}"] for c in LABEL_COLS])

        log.info("train_loss=%.4f  val_loss=%.4f  avg_f1=%.3f",
                 train_loss, metrics["val_loss"], avg_f1)
        for col in LABEL_COLS:
            name = col.replace("label_", "")
            log.info("  %-20s F1=%.3f  PR-AUC=%.3f",
                     name, metrics[f"f1_{name}"], metrics[f"prauc_{name}"])

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), local_model_pt)
            log.info("  >> Saved best model (avg_f1=%.3f)", best_f1)

    # ── Save artifacts ────────────────────────────────────────────
    log.info("Training complete. Best avg F1: %.4f", best_f1)

    meta = {
        "model_name":  args.model_name,
        "label_cols":  LABEL_COLS,
        "max_length":  args.max_length,
        "best_avg_f1": round(best_f1, 4),
        "quick_test":  False,
        "epochs":      args.epochs,
    }
    with open(local_meta, "w") as f:
        json.dump(meta, f, indent=2)

    tokenizer.save_pretrained(local_tok_dir)
    save_artifacts(local_model_pt, local_meta, local_tok_dir, args.output_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
