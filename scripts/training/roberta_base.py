"""Train RoBERTa on EDOS and test on Online Misogyny with the OCR post dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
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
        description="Train RoBERTa on EDOS and evaluate on post_ocr_dataset.csv."
    )
    parser.add_argument(
        "--edos_path",
        type=Path,
        default=Path("datasets/edos/data/edos_labelled_aggregated.csv"),
        help="Path to EDOS aggregated labels CSV.",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        default=Path("datasets/post_ocr_dataset.csv"),
        help="Path to the merged post/OCR dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/roberta_base"),
        help="Directory for metrics, predictions, and the trained model.",
    )
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_ocr_text",
        action="store_true",
        help="Use only raw post text from the test CSV, ignoring OCR text.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_post_text(row: pd.Series, include_ocr_text: bool) -> str:
    parts = [str(row.get("raw_post_text", "")).strip()]
    if include_ocr_text:
        ocr_text = str(row.get("ocr_text", "")).strip()
        if ocr_text and ocr_text not in parts:
            parts.append(ocr_text)
    return "\n".join(part for part in parts if part)


class MisogynyDataset(Dataset):
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


def load_train_val_data(
    edos_path: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load EDOS and create a stratified train/validation split."""
    edos = pd.read_csv(edos_path, keep_default_na=False)
    edos = edos[edos["split"].isin(["train", "dev"])].copy()
    edos["binary_label"] = edos["label_sexist"].eq("sexist").astype(int)
    edos["text"] = edos["text"].map(clean_text)
    edos["source"] = "edos"
    edos = edos[["text", "binary_label", "source"]]
    print(
        f"EDOS train+dev: {len(edos):,} samples "
        f"({int(edos['binary_label'].sum()):,} sexist)"
    )

    data = edos[edos["text"].str.strip().ne("")].copy()
    train_df, val_df = train_test_split(
        data,
        test_size=0.15,
        random_state=seed,
        stratify=data["binary_label"],
    )
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_test_data(test_path: Path, include_ocr_text: bool) -> pd.DataFrame:
    """Load Online Misogyny post/OCR data as a held-out test set."""
    test = pd.read_csv(test_path, keep_default_na=False)
    label_map = {"Nonmisogynistic": 0, "Misogynistic": 1}
    test = test[test["label"].isin(label_map)].copy()
    test["binary_label"] = test["label"].map(label_map).astype(int)
    test["text"] = test.apply(
        lambda row: clean_text(combine_post_text(row, include_ocr_text)),
        axis=1,
    )
    test = test[test["text"].str.strip().ne("")].copy()
    print(
        f"Online Misogyny test: {len(test):,} samples "
        f"({int(test['binary_label'].sum()):,} misogynistic)"
    )
    return test.reset_index(drop=True)


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    weights: torch.Tensor,
) -> tuple[float, float]:
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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs.logits, dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    return total_loss / max(len(dataloader), 1), f1_score(all_labels, all_preds, average="macro")


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
) -> tuple[float, float, list[int], list[int], str]:
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    return total_loss / max(len(dataloader), 1), macro_f1, all_labels, all_preds, report


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")

    train_df, val_df = load_train_val_data(args.edos_path, args.seed)
    test_df = load_test_data(args.test_path, include_ocr_text=not args.no_ocr_text)

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

    train_loader = DataLoader(
        MisogynyDataset(train_df["text"], train_df["binary_label"], tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MisogynyDataset(val_df["text"], val_df["binary_label"], tokenizer, args.max_len),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        MisogynyDataset(test_df["text"], test_df["binary_label"], tokenizer, args.max_len),
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * 0.1), 1),
        num_training_steps=total_steps,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device, weights)
        val_loss, val_f1, _, _, val_report = evaluate(model, val_loader, device, weights)
        print(f"Train loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")
        print(val_report)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
        )

    test_loss, test_f1, test_labels, test_preds, test_report = evaluate(
        model,
        test_loader,
        device,
        weights,
    )
    print("\nHeld-out post_ocr_dataset results")
    print(f"Test loss: {test_loss:.4f} | Test macro F1: {test_f1:.4f}")
    print(test_report)

    predictions = test_df[["post_id", "label", "raw_post_text", "ocr_text", "ocr_readable"]].copy()
    predictions["true_label"] = test_labels
    predictions["predicted_label"] = test_preds
    predictions["predicted_label_text"] = predictions["predicted_label"].map(
        {0: "Nonmisogynistic", 1: "Misogynistic"}
    )
    predictions.to_csv(args.output_dir / "post_ocr_predictions.csv", index=False)

    metrics = {
        "model_name": args.model_name,
        "test_path": str(args.test_path),
        "include_ocr_text": not args.no_ocr_text,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "lr": args.lr,
        "seed": args.seed,
        "test_loss": test_loss,
        "test_macro_f1": test_f1,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.txt").write_text(test_report, encoding="utf-8")

    model_dir = args.output_dir / "best_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Saved predictions, metrics, and model to {args.output_dir}")


if __name__ == "__main__":
    main()
