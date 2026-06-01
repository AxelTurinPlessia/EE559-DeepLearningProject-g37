"""Fine-tune the best RoBERTa checkpoint on synthetic training augmentation.

Synthetic data is used for training only. Real validation and test splits are
kept separate from the synthetic benchmark diagnostics.

Example:
    python scripts/09_finetune_roberta_on_synthetic.py \
      --synthetic_train_path datasets/synthetic/synthetic_train_set.jsonl \
      --base_model_path results/roberta_mixed_post_ocr/lr2e-5_seed42/best_model \
      --output_dir results/roberta_synthetic_finetune \
      --epochs 4 \
      --batch_size 16 \
      --learning_rate 2e-5 \
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import emoji
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


LABEL_NAMES = ["non-misogynistic", "misogynistic"]
LABEL_TO_ID = {
    "non_misogynistic": 0,
    "non-misogynistic": 0,
    "non misogynistic": 0,
    "nonmisogynistic": 0,
    "not_misogynistic": 0,
    "not misogynistic": 0,
    "not sexist": 0,
    "non sexist": 0,
    "non_sexist": 0,
    "misogynistic": 1,
    "sexist": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune current best RoBERTa on synthetic training-only augmentation."
    )
    parser.add_argument(
        "--synthetic_train_path",
        type=Path,
        default=Path("datasets/synthetic/synthetic_train_set.jsonl"),
        help="Synthetic JSONL file used for training only.",
    )
    parser.add_argument(
        "--base_model_path",
        type=Path,
        default=None,
        help="Base checkpoint. Defaults to the best run in results/roberta_mixed_post_ocr.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/roberta_synthetic_finetune"),
        help="Directory for the new checkpoint and metrics.",
    )
    parser.add_argument(
        "--real_val_path",
        type=Path,
        default=Path("datasets/roberta_mixed_post_ocr/val.csv"),
        help="Real validation CSV used for checkpoint selection.",
    )
    parser.add_argument("--edos_path", type=Path, default=Path("datasets/edos/data/edos_labelled_aggregated.csv"))
    parser.add_argument(
        "--hatemoji_test_path",
        type=Path,
        default=Path("datasets/Hatemoji/HatemojiBuild/test.csv"),
    )
    parser.add_argument(
        "--post_ocr_test_path",
        type=Path,
        default=Path("datasets/roberta_mixed_post_ocr/test.csv"),
    )
    parser.add_argument("--synthetic_implicit_path", type=Path, default=Path("datasets/synthetic/implicit.jsonl"))
    parser.add_argument(
        "--synthetic_neutral_path",
        type=Path,
        default=Path("datasets/synthetic/counterexamples.jsonl"),
    )
    parser.add_argument("--synthetic_explicit_path", type=Path, default=Path("datasets/synthetic/explicit.json"))
    parser.add_argument("--synthetic_pairs_path", type=Path, default=Path("datasets/synthetic/pairs.jsonl"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--no_real_validation",
        action="store_true",
        help="Save the final synthetic-finetuned checkpoint instead of selecting by real validation F1.",
    )
    parser.add_argument(
        "--limit_train_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit for synthetic training rows.",
    )
    parser.add_argument(
        "--limit_eval_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit per evaluation dataset.",
    )
    parser.add_argument(
        "--skip_synthetic_benchmark",
        action="store_true",
        help="Skip synthetic benchmark evaluation.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def find_best_roberta_checkpoint() -> Path:
    summary_path = Path("results/roberta_mixed_post_ocr/sweep_summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run = summary.get("best_run", {}).get("run")
        if run:
            checkpoint = Path("results/roberta_mixed_post_ocr") / run / "best_model"
            if (checkpoint / "config.json").exists():
                return checkpoint
    fallback = Path("results/roberta_mixed_post_ocr/lr2e-5_seed42/best_model")
    if (fallback / "config.json").exists():
        return fallback
    raise FileNotFoundError(
        "Could not find a previous RoBERTa checkpoint. Pass --base_model_path explicitly."
    )


def resolve_checkpoint(path: Path | None) -> Path:
    checkpoint = find_best_roberta_checkpoint() if path is None else path
    if (checkpoint / "config.json").exists():
        return checkpoint
    best_model = checkpoint / "best_model"
    if (best_model / "config.json").exists():
        return best_model
    raise FileNotFoundError(f"Missing checkpoint config.json under {checkpoint}")


def normalize_label(value: Any) -> int:
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return int(value)
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    if text in LABEL_TO_ID:
        return LABEL_TO_ID[text]
    raise ValueError(f"Unsupported label: {value!r}")


def process_emojis(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return emoji.demojize(text, delimiters=(" :", ": "))


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = process_emojis(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    require_path(path, "JSON/JSONL dataset")
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: row is not a JSON object")
                rows.append(row)
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "examples", "rows", "items"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def validate_synthetic_train(path: Path, limit: int | None = None) -> pd.DataFrame:
    rows = read_json_or_jsonl(path)
    required = {"id", "text", "label"}
    clean_rows = []
    seen_texts: set[str] = set()
    duplicate_texts = 0
    for idx, row in enumerate(rows, start=1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{path}:{idx} missing required fields: {missing}")
        text = str(row["text"]).strip()
        if not text:
            raise ValueError(f"{path}:{idx} has empty text")
        if text in seen_texts:
            duplicate_texts += 1
        seen_texts.add(text)
        clean_rows.append(
            {
                "sample_id": str(row["id"]),
                "text": text,
                "binary_label": normalize_label(row["label"]),
                "label_text": row["label"],
                "source": "synthetic_train",
            }
        )
    df = pd.DataFrame(clean_rows)
    if limit is not None:
        if set(df["binary_label"].tolist()) == {0, 1} and limit >= 2:
            per_class = max(limit // 2, 1)
            pieces = [
                group.head(per_class)
                for _, group in df.groupby("binary_label", sort=True)
            ]
            df = pd.concat(pieces, ignore_index=True).head(limit).copy()
        else:
            df = df.head(limit).copy()
    print(
        f"Synthetic train: {len(df):,} rows | "
        f"class_counts={df['binary_label'].value_counts().sort_index().to_dict()}"
    )
    if duplicate_texts:
        print(f"WARNING: found {duplicate_texts} duplicated text rows in full synthetic training file.")
    return df.reset_index(drop=True)


class TextDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer, max_len: int) -> None:
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


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    loss_fn,
    desc: str = "Evaluating",
) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    preds_all: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
    return total_loss / max(len(loader), 1), labels_all, preds_all


def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    loss_fn,
    scaler,
    use_fp16: bool,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    labels_all: list[int] = []
    preds_all: list[int] = []
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        labels_all.extend(labels.detach().cpu().tolist())
        preds_all.extend(torch.argmax(outputs.logits, dim=1).detach().cpu().tolist())
    return (
        total_loss / max(len(loader), 1),
        f1_score(labels_all, preds_all, average="macro", zero_division=0),
        accuracy_score(labels_all, preds_all),
    )


def load_real_validation(path: Path, limit: int | None = None) -> pd.DataFrame:
    require_path(path, "real validation CSV")
    df = pd.read_csv(path, keep_default_na=False)
    required = {"text", "binary_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df[df["text"].astype(str).str.strip().ne("")].copy()
    out["binary_label"] = out["binary_label"].astype(int)
    if "source" not in out.columns:
        out["source"] = "real_val"
    if limit is not None:
        out = out.head(limit).copy()
    print(
        f"Real validation: {len(out):,} rows | "
        f"class_counts={out['binary_label'].value_counts().sort_index().to_dict()}"
    )
    return out.reset_index(drop=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_edos_test(path: Path) -> pd.DataFrame:
    require_path(path, "EDOS CSV")
    rows = []
    for row in read_csv_rows(path):
        if row["split"] != "test":
            continue
        rows.append(
            {
                "sample_id": f"edos_{row['rewire_id']}",
                "source": "edos_test",
                "text": clean_text(row["text"]),
                "binary_label": int(row["label_sexist"] == "sexist"),
                "label_text": "misogynistic" if row["label_sexist"] == "sexist" else "non-misogynistic",
            }
        )
    return pd.DataFrame(rows)


def load_hatemoji_test(path: Path) -> pd.DataFrame:
    require_path(path, "Hatemoji test CSV")
    rows = []
    for row in read_csv_rows(path):
        rows.append(
            {
                "sample_id": f"hatemoji_{row['entry_id']}",
                "source": "hatemoji_test",
                "text": clean_text(row["text"]),
                "binary_label": int(row["label_gold"]),
                "label_text": "misogynistic" if int(row["label_gold"]) == 1 else "non-misogynistic",
            }
        )
    return pd.DataFrame(rows)


def load_post_ocr_test(path: Path) -> pd.DataFrame:
    require_path(path, "post/OCR held-out CSV")
    df = pd.read_csv(path, keep_default_na=False)
    required = {"text", "binary_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df.copy()
    out["source"] = "post_ocr_test"
    out["binary_label"] = out["binary_label"].astype(int)
    if "sample_id" not in out.columns:
        out["sample_id"] = [f"post_ocr_{idx}" for idx in range(len(out))]
    return out


def load_real_world_tests(args: argparse.Namespace) -> pd.DataFrame:
    frames = [
        load_edos_test(args.edos_path),
        load_hatemoji_test(args.hatemoji_test_path),
        load_post_ocr_test(args.post_ocr_test_path),
    ]
    out = pd.concat(frames, ignore_index=True)
    out = out[out["text"].astype(str).str.strip().ne("")].copy()
    if args.limit_eval_samples is not None:
        out = (
            out.groupby("source", group_keys=False)
            .head(args.limit_eval_samples)
            .reset_index(drop=True)
        )
    return out.reset_index(drop=True)


def load_json_benchmark(path: Path, source: str) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(read_json_or_jsonl(path), start=1):
        rows.append(
            {
                "sample_id": str(row.get("id", f"{source}_{idx:04d}")),
                "source": source,
                "text": str(row.get("text", "")),
                "binary_label": normalize_label(row.get("label", "")),
                "label_text": str(row.get("label", "")),
            }
        )
    return pd.DataFrame(rows)


def load_synthetic_pairs(path: Path) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(read_json_or_jsonl(path), start=1):
        y = normalize_label(row.get("label", ""))
        pair_id = str(row.get("base_pair_id", f"pair_{idx:04d}"))
        for condition, key in [("plain", "plain_text"), ("emoji", "emoji_text")]:
            rows.append(
                {
                    "sample_id": f"{pair_id}_{condition}",
                    "source": f"synthetic_pairs_{condition}",
                    "text": str(row.get(key, "")),
                    "binary_label": y,
                    "label_text": str(row.get("label", "")),
                }
            )
    return pd.DataFrame(rows)


def load_synthetic_benchmarks(args: argparse.Namespace) -> pd.DataFrame:
    frames = [
        load_json_benchmark(args.synthetic_implicit_path, "synthetic_implicit"),
        load_json_benchmark(args.synthetic_neutral_path, "synthetic_neutral"),
        load_json_benchmark(args.synthetic_explicit_path, "synthetic_explicit"),
        load_synthetic_pairs(args.synthetic_pairs_path),
    ]
    out = pd.concat(frames, ignore_index=True)
    out = out[out["text"].astype(str).str.strip().ne("")].copy()
    if args.limit_eval_samples is not None:
        out = (
            out.groupby("source", group_keys=False)
            .head(args.limit_eval_samples)
            .reset_index(drop=True)
        )
    return out.reset_index(drop=True)


def metric_payload(df: pd.DataFrame, labels: list[int], preds: list[int]) -> dict[str, Any]:
    report = classification_report(
        labels,
        preds,
        target_names=LABEL_NAMES,
        labels=[0, 1],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    per_source = {}
    eval_df = df.copy()
    eval_df["true_label"] = labels
    eval_df["predicted_label"] = preds
    for source, group in eval_df.groupby("source"):
        per_source[source] = {
            "n": int(len(group)),
            "class_distribution": {
                LABEL_NAMES[int(label)]: int(count)
                for label, count in group["binary_label"].value_counts().sort_index().items()
            },
            "accuracy": float(accuracy_score(group["true_label"], group["predicted_label"])),
            "macro_f1": float(
                f1_score(group["true_label"], group["predicted_label"], average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(group["true_label"], group["predicted_label"], average="weighted", zero_division=0)
            ),
        }
    return {
        "n_examples": int(len(df)),
        "label_mapping": {"non_misogynistic": 0, "misogynistic": 1},
        "class_distribution": {
            LABEL_NAMES[int(label)]: int(count)
            for label, count in df["binary_label"].value_counts().sort_index().items()
        },
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "classification_report": report,
        "per_source": per_source,
    }


def save_eval_outputs(
    prefix: str,
    output_dir: Path,
    df: pd.DataFrame,
    labels: list[int],
    preds: list[int],
) -> dict[str, Any]:
    payload = metric_payload(df, labels, preds)
    (output_dir / f"metrics_{prefix}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    report_text = classification_report(
        labels,
        preds,
        target_names=LABEL_NAMES,
        labels=[0, 1],
        digits=4,
        zero_division=0,
    )
    (output_dir / f"classification_report_{prefix}.txt").write_text(report_text, encoding="utf-8")
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        output_dir / f"confusion_matrix_{prefix}.csv"
    )
    pred_df = df.copy()
    pred_df["true_label"] = labels
    pred_df["predicted_label"] = preds
    pred_df["predicted_label_text"] = pred_df["predicted_label"].map(
        {0: LABEL_NAMES[0], 1: LABEL_NAMES[1]}
    )
    pred_df.to_csv(output_dir / f"predictions_{prefix}.csv", index=False)
    return payload


def check_synthetic_overlap(train_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> None:
    overlap = set(train_df["text"]) & set(benchmark_df["text"])
    if overlap:
        print(
            "WARNING: synthetic benchmark has exact text overlap with synthetic training set: "
            f"{len(overlap)} unique texts."
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = resolve_checkpoint(args.base_model_path)
    print(f"Base checkpoint: {base_model_path}")
    print(f"Output directory: {args.output_dir}")

    train_df = validate_synthetic_train(args.synthetic_train_path, args.limit_train_samples)
    val_df = None if args.no_real_validation else load_real_validation(args.real_val_path, args.limit_eval_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    print(f"Device: {device}")
    print(f"Using fp16: {use_fp16}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=2)
    model.to(device)

    train_labels = train_df["binary_label"].to_numpy()
    if set(train_labels.tolist()) == {0, 1}:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=train_labels,
        )
    else:
        class_weights = np.array([1.0, 1.0])
        print(
            "WARNING: training subset does not contain both classes; "
            "using unit class weights. This is expected for some tiny smoke tests."
        )
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    print(f"Training class weights: {class_weights}")

    train_loader = make_loader(
        train_df,
        tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_df is not None:
        val_loader = make_loader(
            val_df,
            tokenizer,
            args.max_length,
            args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * args.warmup_ratio), 1),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16) if use_fp16 else None

    checkpoint_dir = args.output_dir / "best_model"
    final_dir = args.output_dir / "final_model"
    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_macro_f1, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            loss_fn,
            scaler,
            use_fp16,
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_macro_f1,
            "train_accuracy": train_acc,
        }
        print(f"Train loss: {train_loss:.4f} | macro F1: {train_macro_f1:.4f} | acc: {train_acc:.4f}")

        if val_loader is not None:
            val_loss, val_labels, val_preds = evaluate(model, val_loader, device, loss_fn, desc="Real validation")
            val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(val_labels, val_preds)
            improved = val_macro_f1 > best_val_f1
            if improved:
                best_val_f1 = val_macro_f1
                best_epoch = epoch
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"Saved new best real-validation checkpoint to {checkpoint_dir}")
            print(f"Val loss: {val_loss:.4f} | macro F1: {val_macro_f1:.4f} | acc: {val_acc:.4f}")
            record.update(
                {
                    "val_loss": val_loss,
                    "val_macro_f1": val_macro_f1,
                    "val_accuracy": val_acc,
                    "is_best": improved,
                }
            )
        history.append(record)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if val_loader is None:
        checkpoint_dir = final_dir
        best_epoch = args.epochs
        print(f"Saved final checkpoint to {final_dir}")
    else:
        print(f"\nReloading best checkpoint from epoch {best_epoch}: {checkpoint_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=2)
        model.to(device)

    train_metadata = {
        "base_model_path": str(base_model_path),
        "synthetic_train_path": str(args.synthetic_train_path),
        "output_dir": str(args.output_dir),
        "label_mapping": {"non_misogynistic": 0, "misogynistic": 1},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "fp16": use_fp16,
        "selected_checkpoint": str(checkpoint_dir),
        "best_epoch": best_epoch,
        "history": history,
        "synthetic_train_class_counts": {
            LABEL_NAMES[int(label)]: int(count)
            for label, count in train_df["binary_label"].value_counts().sort_index().items()
        },
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(train_metadata, indent=2),
        encoding="utf-8",
    )

    eval_loss_fn = torch.nn.CrossEntropyLoss()
    real_world_df = load_real_world_tests(args)
    print(
        f"\nReal-world test: {len(real_world_df):,} rows | "
        f"sources={real_world_df['source'].value_counts().sort_index().to_dict()}"
    )
    real_loader = make_loader(
        real_world_df,
        tokenizer,
        args.max_length,
        args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    _, real_labels, real_preds = evaluate(model, real_loader, device, eval_loss_fn, desc="Real-world test")
    real_metrics = save_eval_outputs("real_world", args.output_dir, real_world_df, real_labels, real_preds)
    print(
        "Real-world metrics: "
        f"accuracy={real_metrics['accuracy']:.4f} | "
        f"macro_f1={real_metrics['macro_f1']:.4f} | "
        f"weighted_f1={real_metrics['weighted_f1']:.4f}"
    )

    if not args.skip_synthetic_benchmark:
        synthetic_benchmark_df = load_synthetic_benchmarks(args)
        check_synthetic_overlap(train_df, synthetic_benchmark_df)
        print(
            f"\nSynthetic benchmark: {len(synthetic_benchmark_df):,} rows | "
            f"sources={synthetic_benchmark_df['source'].value_counts().sort_index().to_dict()}"
        )
        synthetic_loader = make_loader(
            synthetic_benchmark_df,
            tokenizer,
            args.max_length,
            args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        _, syn_labels, syn_preds = evaluate(
            model,
            synthetic_loader,
            device,
            eval_loss_fn,
            desc="Synthetic benchmark",
        )
        syn_metrics = save_eval_outputs(
            "synthetic_benchmarks",
            args.output_dir,
            synthetic_benchmark_df,
            syn_labels,
            syn_preds,
        )
        print(
            "Synthetic benchmark metrics: "
            f"accuracy={syn_metrics['accuracy']:.4f} | "
            f"macro_f1={syn_metrics['macro_f1']:.4f} | "
            f"weighted_f1={syn_metrics['weighted_f1']:.4f}"
        )

    print(f"\nSaved checkpoint and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
