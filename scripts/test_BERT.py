"""Evaluate a fine-tuned BERT model on the merged post/OCR dataset."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_NAMES = ["non-misogynistic", "misogynistic"]
DEFAULT_CHECKPOINT_CANDIDATES = [
    Path("results/bert/model"),
    Path("results/bert/best_model"),
    Path("results_full/best_model"),
    Path("/scratch/results_full/best_model"),
    Path("/scratch/bert_model"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned BERT classifier on datasets/post_ocr_dataset.csv."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Path to a fine-tuned BERT checkpoint/model directory.",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        default=Path("datasets/post_ocr_dataset.csv"),
        help="Path to the merged post/OCR dataset used by the RoBERTa test.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/bert_post_ocr"),
        help="Directory where BERT post/OCR evaluation outputs are written.",
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--no_ocr_text",
        action="store_true",
        help="Use only raw post text, ignoring OCR text.",
    )
    return parser.parse_args()


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {path}" for path in DEFAULT_CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        "No BERT checkpoint/model directory found. Pass one explicitly with "
        f"--model_path.\nSearched:\n{searched}"
    )


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def combine_post_text(row: pd.Series, include_ocr_text: bool) -> str:
    parts = [str(row.get("raw_post_text", "")).strip()]
    if include_ocr_text:
        ocr_text = str(row.get("ocr_text", "")).strip()
        if ocr_text and ocr_text not in parts:
            parts.append(ocr_text)
    return "\n".join(part for part in parts if part)


def load_post_ocr_data(test_path: Path, include_ocr_text: bool) -> pd.DataFrame:
    test = pd.read_csv(test_path, keep_default_na=False)
    label_map = {"Nonmisogynistic": 0, "Misogynistic": 1}
    test = test[test["label"].isin(label_map)].copy()
    test["binary_label"] = test["label"].map(label_map).astype(int)
    test["text"] = test.apply(
        lambda row: clean_text(combine_post_text(row, include_ocr_text)),
        axis=1,
    )
    test = test[test["text"].str.strip().ne("")].copy()
    return test.reset_index(drop=True)


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


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, list[int], list[int], str]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating BERT"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    return total_loss / max(len(dataloader), 1), accuracy, macro_f1, all_labels, all_preds, report


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    include_ocr_text = not args.no_ocr_text
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using BERT model: {model_path}")
    print(f"Testing on: {args.test_path}")
    print(f"Include OCR text: {include_ocr_text}")

    test_df = load_post_ocr_data(args.test_path, include_ocr_text=include_ocr_text)
    print(
        f"Loaded {len(test_df):,} post/OCR test rows "
        f"({int(test_df['binary_label'].sum()):,} misogynistic)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    test_loader = DataLoader(
        TextDataset(test_df["text"], test_df["binary_label"], tokenizer, args.max_len),
        batch_size=args.batch_size,
    )
    test_loss, accuracy, macro_f1, test_labels, test_preds, report = evaluate(
        model,
        test_loader,
        device,
    )

    print("\nHeld-out post_ocr_dataset results")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)

    predictions = test_df[["post_id", "label", "raw_post_text", "ocr_text", "ocr_readable"]].copy()
    predictions["true_label"] = test_labels
    predictions["predicted_label"] = test_preds
    predictions["predicted_label_text"] = predictions["predicted_label"].map(
        {0: "Nonmisogynistic", 1: "Misogynistic"}
    )
    predictions.to_csv(args.output_dir / "post_ocr_predictions.csv", index=False)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "test_path": str(args.test_path),
        "include_ocr_text": include_ocr_text,
        "num_examples": int(len(test_df)),
        "test_loss": float(test_loss),
        "accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "label_counts": {
            label: int(count)
            for label, count in test_df["binary_label"].value_counts().sort_index().items()
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print(f"Saved predictions, metrics, and report to {args.output_dir}")


if __name__ == "__main__":
    main()
