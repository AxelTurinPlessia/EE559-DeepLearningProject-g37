"""Fine-tune RoBERTa on mixed RoBERTa/post-OCR splits.

This script expects split CSVs created by create_mixed_roberta_dataset.py and
keeps the final held-out test evaluation separate from checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


LABEL_NAMES = ["non-misogynistic", "misogynistic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a RoBERTa checkpoint on mixed train/val/test CSVs."
    )
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=Path("datasets/roberta_mixed_post_ocr"),
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/roberta_mixed_post_ocr"),
        help="Directory for metrics, predictions, and checkpoints.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="results/roberta_emoji/model",
        help="Model name or local checkpoint path to fine-tune.",
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Stop after this many epochs without validation macro-F1 improvement.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use CUDA mixed precision training when a GPU is available.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer, max_len: int):
        self.texts = texts.astype(str).tolist()
        self.labels = labels.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_split(split_dir: Path, split: str) -> pd.DataFrame:
    path = split_dir / f"{split}.csv"
    df = pd.read_csv(path, keep_default_na=False)
    required = {"text", "binary_label", "source"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[df["text"].astype(str).str.strip().ne("")].copy()
    df["binary_label"] = df["binary_label"].astype(int)
    print(
        f"{split}: {len(df):,} rows | "
        f"positive={int(df['binary_label'].sum()):,} | "
        f"sources={df['source'].value_counts().sort_index().to_dict()}"
    )
    return df.reset_index(drop=True)


def make_loader(
    df: pd.DataFrame,
    tokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TextDataset(df["text"], df["binary_label"], tokenizer, max_len),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    weights: torch.Tensor,
    scaler,
    use_fp16: bool,
) -> tuple[float, float, float]:
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

        if scaler is not None:
            scale_before_step = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before_step:
                scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs.logits, dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / max(len(dataloader), 1), macro_f1, accuracy


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
) -> tuple[float, float, float, list[int], list[int], str]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    return total_loss / max(len(dataloader), 1), macro_f1, accuracy, all_labels, all_preds, report


def save_predictions(
    path: Path,
    df: pd.DataFrame,
    true_labels: list[int],
    predictions: list[int],
) -> None:
    prediction_columns = [
        column
        for column in [
            "sample_id",
            "split",
            "source",
            "source_split",
            "post_id",
            "raw_post_text",
            "ocr_text",
            "ocr_readable",
            "text",
            "binary_label",
            "label_text",
        ]
        if column in df.columns
    ]
    output = df[prediction_columns].copy()
    output["true_label"] = true_labels
    output["predicted_label"] = predictions
    output["predicted_label_text"] = output["predicted_label"].map(
        {0: LABEL_NAMES[0], 1: LABEL_NAMES[1]}
    )
    output.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"
    print(f"Using device: {device}")
    print(f"Using fp16: {use_fp16}")
    print(f"Loading model: {args.model_name}")

    train_df = load_split(args.split_dir, "train")
    val_df = load_split(args.split_dir, "val")
    test_df = load_split(args.split_dir, "test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["binary_label"].to_numpy(),
    )
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights}")

    train_loader = make_loader(
        train_df,
        tokenizer,
        args.max_len,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_df,
        tokenizer,
        args.max_len,
        args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_df,
        tokenizer,
        args.max_len,
        args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * args.warmup_ratio), 1),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16) if use_fp16 else None

    history: list[dict[str, float | int]] = []
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_dir = args.output_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_f1, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            weights,
            scaler,
            use_fp16,
        )
        val_loss, val_f1, val_accuracy, _, _, val_report = evaluate(
            model,
            val_loader,
            device,
            weights,
        )

        print(f"Train loss: {train_loss:.4f} | macro F1: {train_f1:.4f} | acc: {train_accuracy:.4f}")
        print(f"Val   loss: {val_loss:.4f} | macro F1: {val_f1:.4f} | acc: {val_accuracy:.4f}")
        print(val_report)

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"Saved new best checkpoint to {best_model_dir}")
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "val_accuracy": val_accuracy,
                "is_best": improved,
            }
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without improvement.")
            break

    print(f"\nReloading best checkpoint from epoch {best_epoch}: {best_model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir, num_labels=2)
    model.to(device)

    val_loss, val_f1, val_accuracy, val_labels, val_preds, val_report = evaluate(
        model,
        val_loader,
        device,
        weights,
    )
    test_loss, test_f1, test_accuracy, test_labels, test_preds, test_report = evaluate(
        model,
        test_loader,
        device,
        weights,
    )

    print("\nBest-checkpoint validation results")
    print(f"Val loss: {val_loss:.4f} | macro F1: {val_f1:.4f} | acc: {val_accuracy:.4f}")
    print(val_report)
    print("\nHeld-out post/OCR test results")
    print(f"Test loss: {test_loss:.4f} | macro F1: {test_f1:.4f} | acc: {test_accuracy:.4f}")
    print(test_report)

    save_predictions(args.output_dir / "val_predictions.csv", val_df, val_labels, val_preds)
    save_predictions(args.output_dir / "test_predictions.csv", test_df, test_labels, test_preds)
    (args.output_dir / "validation_report.txt").write_text(val_report, encoding="utf-8")
    (args.output_dir / "classification_report.txt").write_text(test_report, encoding="utf-8")

    metrics = {
        "model_name": args.model_name,
        "split_dir": str(args.split_dir),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1_during_training": best_val_f1,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_len": args.max_len,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "fp16": use_fp16,
        "history": history,
        "best_checkpoint_validation": {
            "loss": val_loss,
            "macro_f1": val_f1,
            "accuracy": val_accuracy,
        },
        "test": {
            "loss": test_loss,
            "macro_f1": test_f1,
            "accuracy": test_accuracy,
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Saved metrics, predictions, reports, and best model to {args.output_dir}")


if __name__ == "__main__":
    main()
