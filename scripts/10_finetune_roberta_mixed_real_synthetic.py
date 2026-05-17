"""Fine-tune RoBERTa on an 80/20 real + synthetic training mixture.

The synthetic data is used only as training augmentation. Checkpoint selection
and optional threshold tuning use the real validation split only.

Example:
    python scripts/10_finetune_roberta_mixed_real_synthetic.py \
      --base_model_path results/roberta_mixed_post_ocr/lr2e-5_seed42/best_model \
      --synthetic_train_path datasets/synthetic/synthetic_train_set.jsonl \
      --synthetic_ratio 0.20 \
      --synthetic_sampling auto \
      --output_dir results/roberta_mixed_real80_synthetic20_lr5e-6_seed42 \
      --epochs 1 \
      --batch_size 16 \
      --learning_rate 5e-6 \
      --seed 42 \
      --tune_threshold
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
        description="Fine-tune RoBERTa with real training data plus synthetic augmentation."
    )
    parser.add_argument(
        "--base_model_path",
        type=Path,
        default=Path("results/roberta_mixed_post_ocr/lr2e-5_seed42/best_model"),
    )
    parser.add_argument("--real_train_path", type=Path, default=Path("datasets/roberta_mixed_post_ocr/train.csv"))
    parser.add_argument("--real_val_path", type=Path, default=Path("datasets/roberta_mixed_post_ocr/val.csv"))
    parser.add_argument(
        "--synthetic_train_path",
        type=Path,
        default=Path("datasets/synthetic/synthetic_train_set.jsonl"),
    )
    parser.add_argument("--synthetic_ratio", type=float, default=0.20)
    parser.add_argument(
        "--synthetic_sampling",
        choices=["without_replacement", "with_replacement", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/roberta_mixed_real80_synthetic20_lr5e-6_seed42"),
    )
    parser.add_argument("--edos_path", type=Path, default=Path("datasets/edos/data/edos_labelled_aggregated.csv"))
    parser.add_argument("--hatemoji_test_path", type=Path, default=Path("datasets/Hatemoji/HatemojiBuild/test.csv"))
    parser.add_argument("--post_ocr_test_path", type=Path, default=Path("datasets/roberta_mixed_post_ocr/test.csv"))
    parser.add_argument("--synthetic_implicit_path", type=Path, default=Path("datasets/synthetic/implicit.jsonl"))
    parser.add_argument("--synthetic_neutral_path", type=Path, default=Path("datasets/synthetic/counterexamples.jsonl"))
    parser.add_argument("--synthetic_explicit_path", type=Path, default=Path("datasets/synthetic/explicit.json"))
    parser.add_argument("--synthetic_pairs_path", type=Path, default=Path("datasets/synthetic/pairs.jsonl"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--limit_real_train_samples", type=int, default=None)
    parser.add_argument("--limit_synthetic_train_samples", type=int, default=None)
    parser.add_argument("--limit_eval_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {path}. "
            "Use --overwrite for smoke tests or choose a new output_dir."
        )
    path.mkdir(parents=True, exist_ok=True)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def resolve_checkpoint(path: Path) -> Path:
    if (path / "config.json").exists():
        return path
    best_model = path / "best_model"
    if (best_model / "config.json").exists():
        return best_model
    raise FileNotFoundError(f"Missing checkpoint config.json under {path}")


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
    return re.sub(r"\s+", " ", text).strip()


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        LABEL_NAMES[int(label)]: int(count)
        for label, count in df["binary_label"].value_counts().sort_index().items()
    }


def load_real_split(path: Path, split_name: str, limit: int | None = None) -> pd.DataFrame:
    require_path(path, f"real {split_name} CSV")
    df = pd.read_csv(path, keep_default_na=False)
    required = {"text", "binary_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df[df["text"].astype(str).str.strip().ne("")].copy()
    out["binary_label"] = out["binary_label"].astype(int)
    if "source" not in out.columns:
        out["source"] = f"real_{split_name}"
    out["origin"] = "real"
    if limit is not None:
        out = (
            out.groupby("binary_label", group_keys=False)
            .head(max(limit // 2, 1))
            .head(limit)
            .copy()
        )
    print(f"Real {split_name}: {len(out):,} rows | class_counts={class_counts(out)}")
    return out.reset_index(drop=True)


def load_synthetic_train(path: Path, limit: int | None = None) -> pd.DataFrame:
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
                "label_text": str(row["label"]),
                "source": "synthetic_train",
                "origin": "synthetic",
            }
        )
    df = pd.DataFrame(clean_rows)
    if limit is not None:
        df = (
            df.groupby("binary_label", group_keys=False)
            .head(max(limit // 2, 1))
            .head(limit)
            .copy()
        )
    print(f"Synthetic train pool: {len(df):,} rows | class_counts={class_counts(df)}")
    if duplicate_texts:
        print(f"WARNING: found {duplicate_texts} duplicated text rows in full synthetic training file.")
    return df.reset_index(drop=True)


def select_synthetic_examples(
    synthetic_pool: pd.DataFrame,
    real_n: int,
    ratio: float,
    sampling: str,
    seed: int,
) -> pd.DataFrame:
    if ratio <= 0 or ratio >= 1:
        raise ValueError(f"--synthetic_ratio must be in (0, 1), got {ratio}")
    target_n = int(round(real_n * ratio / (1.0 - ratio)))
    if target_n <= 0:
        raise ValueError("Synthetic target sample count is zero; increase --synthetic_ratio or real data limit.")
    replace = sampling == "with_replacement" or (sampling == "auto" and target_n > len(synthetic_pool))
    if sampling == "without_replacement" and target_n > len(synthetic_pool):
        raise ValueError(
            f"Need {target_n} synthetic rows but pool has {len(synthetic_pool)}. "
            "Use --synthetic_sampling auto or with_replacement."
        )

    pieces = []
    rng = np.random.default_rng(seed)
    label_counts = synthetic_pool["binary_label"].value_counts(normalize=True).sort_index()
    allocated = 0
    for i, (label, proportion) in enumerate(label_counts.items()):
        if i == len(label_counts) - 1:
            n_label = target_n - allocated
        else:
            n_label = int(round(target_n * proportion))
            allocated += n_label
        group = synthetic_pool[synthetic_pool["binary_label"].eq(label)]
        random_state = int(rng.integers(0, 2**31 - 1))
        pieces.append(group.sample(n=n_label, replace=replace, random_state=random_state))
    selected = pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"Selected synthetic train: {len(selected):,} rows | replace={replace} | "
        f"class_counts={class_counts(selected)}"
    )
    return selected


def build_mixed_train(real_train: pd.DataFrame, selected_synthetic: pd.DataFrame, seed: int) -> pd.DataFrame:
    mixed = pd.concat([real_train, selected_synthetic], ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    synthetic_share = float(mixed["origin"].eq("synthetic").mean())
    print(
        f"Mixed train: {len(mixed):,} rows | synthetic_share={synthetic_share:.4f} | "
        f"class_counts={class_counts(mixed)}"
    )
    return mixed


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


def predict_proba(
    model,
    loader: DataLoader,
    device: torch.device,
    loss_fn,
    desc: str,
) -> tuple[float, list[int], list[float]]:
    model.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    probs_all: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            total_loss += loss_fn(outputs.logits, labels).item()
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())
    return total_loss / max(len(loader), 1), labels_all, probs_all


def preds_at_threshold(probs: list[float], threshold: float) -> list[int]:
    return [int(prob >= threshold) for prob in probs]


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


def tune_threshold(labels: list[int], probs: list[float]) -> dict[str, Any]:
    best = {"threshold": 0.5, "macro_f1": -1.0}
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        preds = preds_at_threshold(probs, float(threshold))
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        rows.append({"threshold": float(threshold), "macro_f1": float(macro_f1)})
        if macro_f1 > best["macro_f1"]:
            best = {"threshold": float(threshold), "macro_f1": float(macro_f1)}
    best["grid"] = rows
    return best


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
                "origin": "real_test",
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
                "origin": "real_test",
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
    out["origin"] = "real_test"
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
        out = out.groupby("source", group_keys=False).head(args.limit_eval_samples).reset_index(drop=True)
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
                "origin": "synthetic_benchmark",
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
                    "origin": "synthetic_benchmark",
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
        out = out.groupby("source", group_keys=False).head(args.limit_eval_samples).reset_index(drop=True)
    return out.reset_index(drop=True)


def assert_safety(
    real_val: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic_train: pd.DataFrame,
    synthetic_benchmark: pd.DataFrame,
) -> None:
    if real_val["origin"].eq("synthetic").any():
        raise AssertionError("Validation data contains synthetic examples.")
    if real_test["origin"].ne("real_test").any():
        raise AssertionError("Real-world test data contains non-real-test examples.")
    overlap = set(synthetic_train["text"]) & set(synthetic_benchmark["text"])
    if overlap:
        raise AssertionError(
            "Synthetic benchmark overlaps synthetic training by exact text: "
            f"{len(overlap)} unique texts."
        )


def metric_payload(df: pd.DataFrame, labels: list[int], probs: list[float], threshold: float) -> dict[str, Any]:
    preds = preds_at_threshold(probs, threshold)
    report = classification_report(
        labels,
        preds,
        target_names=LABEL_NAMES,
        labels=[0, 1],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    eval_df = df.copy()
    eval_df["true_label"] = labels
    eval_df["predicted_label"] = preds
    eval_df["p_misogynistic"] = probs
    per_source = {}
    for source, group in eval_df.groupby("source"):
        per_source[source] = {
            "n": int(len(group)),
            "class_distribution": class_counts(group),
            "positive_prediction_rate": float(group["predicted_label"].mean()),
            "accuracy": float(accuracy_score(group["true_label"], group["predicted_label"])),
            "macro_f1": float(f1_score(group["true_label"], group["predicted_label"], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(group["true_label"], group["predicted_label"], average="weighted", zero_division=0)),
        }
    return {
        "threshold": threshold,
        "n_examples": int(len(df)),
        "label_mapping": {"non_misogynistic": 0, "misogynistic": 1},
        "class_distribution": class_counts(df),
        "positive_prediction_rate": float(np.mean(preds)) if preds else 0.0,
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
    probs: list[float],
    threshold: float,
) -> dict[str, Any]:
    tag = "threshold_0p5" if threshold == 0.5 else "threshold_tuned"
    preds = preds_at_threshold(probs, threshold)
    payload = metric_payload(df, labels, probs, threshold)
    (output_dir / f"metrics_{prefix}_{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_text = classification_report(
        labels,
        preds,
        target_names=LABEL_NAMES,
        labels=[0, 1],
        digits=4,
        zero_division=0,
    )
    (output_dir / f"classification_report_{prefix}_{tag}.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(confusion_matrix(labels, preds, labels=[0, 1]), index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        output_dir / f"confusion_matrix_{prefix}_{tag}.csv"
    )
    pred_df = df.copy()
    pred_df["true_label"] = labels
    pred_df["p_misogynistic"] = probs
    pred_df["predicted_label"] = preds
    pred_df["predicted_label_text"] = pred_df["predicted_label"].map({0: LABEL_NAMES[0], 1: LABEL_NAMES[1]})
    pred_df.to_csv(output_dir / f"predictions_{prefix}_{tag}.csv", index=False)
    return payload


def synthetic_diagnostics(df: pd.DataFrame, probs: list[float], threshold: float) -> dict[str, float]:
    preds = preds_at_threshold(probs, threshold)
    tmp = df.copy()
    tmp["predicted_label"] = preds
    tmp["p_misogynistic"] = probs
    out: dict[str, float] = {}
    for source, key in [
        ("synthetic_explicit", "explicit_recall"),
        ("synthetic_implicit", "implicit_recall"),
    ]:
        subset = tmp[tmp["source"].eq(source)]
        out[key] = float(subset["predicted_label"].eq(1).mean()) if len(subset) else float("nan")
    neutral = tmp[tmp["source"].eq("synthetic_neutral")]
    out["neutral_false_positive_rate"] = float(neutral["predicted_label"].eq(1).mean()) if len(neutral) else float("nan")
    plain = tmp[tmp["source"].eq("synthetic_pairs_plain")].copy()
    emoji_df = tmp[tmp["source"].eq("synthetic_pairs_emoji")].copy()
    plain["base_pair_id"] = plain["sample_id"].str.replace("_plain$", "", regex=True)
    emoji_df["base_pair_id"] = emoji_df["sample_id"].str.replace("_emoji$", "", regex=True)
    pair = plain[["base_pair_id", "predicted_label", "p_misogynistic"]].merge(
        emoji_df[["base_pair_id", "predicted_label", "p_misogynistic"]],
        on="base_pair_id",
        suffixes=("_plain", "_emoji"),
    )
    out["emoji_flip_rate"] = float((pair["predicted_label_plain"] != pair["predicted_label_emoji"]).mean()) if len(pair) else float("nan")
    out["emoji_mean_delta_p"] = float((pair["p_misogynistic_emoji"] - pair["p_misogynistic_plain"]).mean()) if len(pair) else float("nan")
    return out


def load_existing_metric(path: Path, synthetic_diag_path: Path | None = None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "real_world_macro_f1": data.get("macro_f1"),
        "real_world_accuracy": data.get("accuracy"),
        "real_world_positive_prediction_rate": data.get("positive_prediction_rate"),
        "real_world_misogynistic_precision": data.get("classification_report", {}).get("misogynistic", {}).get("precision"),
        "real_world_misogynistic_recall": data.get("classification_report", {}).get("misogynistic", {}).get("recall"),
        "edos_macro_f1": data.get("per_source", {}).get("edos_test", {}).get("macro_f1"),
        "post_ocr_macro_f1": data.get("per_source", {}).get("post_ocr_test", {}).get("macro_f1"),
    }
    if synthetic_diag_path and synthetic_diag_path.exists():
        syn = json.loads(synthetic_diag_path.read_text(encoding="utf-8"))
        row["synthetic_benchmark_macro_f1"] = syn.get("macro_f1")
    return row


def write_model_comparison(output_dir: Path, new_real: dict[str, Any], new_synthetic: dict[str, Any]) -> None:
    rows = []
    previous_best = load_existing_metric(Path("results/roberta_synthetic_finetune/metrics_real_world.json"))
    if previous_best:
        previous_best["model"] = "synthetic_only_roberta"
        previous_best["synthetic_benchmark_macro_f1"] = json.loads(
            Path("results/roberta_synthetic_finetune/metrics_synthetic_benchmarks.json").read_text(encoding="utf-8")
        ).get("macro_f1")
        previous_best["implicit_misogyny_recall_synthetic"] = None
        rows.append(previous_best)
    mixed_baseline = Path("results/roberta_mixed_post_ocr/lr2e-5_seed42/metrics.json")
    if mixed_baseline.exists():
        data = json.loads(mixed_baseline.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": "original_mixed_roberta",
                "real_world_macro_f1": None,
                "real_world_accuracy": None,
                "real_world_misogynistic_precision": None,
                "real_world_misogynistic_recall": None,
                "real_world_positive_prediction_rate": None,
                "post_ocr_macro_f1": data.get("test", {}).get("macro_f1"),
                "edos_macro_f1": None,
                "synthetic_benchmark_macro_f1": None,
                "implicit_misogyny_recall_synthetic": None,
            }
        )
    diagnostics = synthetic_diagnostics_from_metric(new_synthetic)
    rows.append(
        {
            "model": "mixed_real80_synthetic20_roberta",
            "real_world_macro_f1": new_real.get("macro_f1"),
            "real_world_accuracy": new_real.get("accuracy"),
            "real_world_misogynistic_precision": new_real.get("classification_report", {}).get("misogynistic", {}).get("precision"),
            "real_world_misogynistic_recall": new_real.get("classification_report", {}).get("misogynistic", {}).get("recall"),
            "real_world_positive_prediction_rate": new_real.get("positive_prediction_rate"),
            "post_ocr_macro_f1": new_real.get("per_source", {}).get("post_ocr_test", {}).get("macro_f1"),
            "edos_macro_f1": new_real.get("per_source", {}).get("edos_test", {}).get("macro_f1"),
            "synthetic_benchmark_macro_f1": new_synthetic.get("macro_f1"),
            "implicit_misogyny_recall_synthetic": diagnostics.get("implicit_recall"),
        }
    )
    pd.DataFrame(rows).to_csv(output_dir / "model_comparison.csv", index=False)


def synthetic_diagnostics_from_metric(metric: dict[str, Any]) -> dict[str, Any]:
    per_source = metric.get("per_source", {})
    return {
        "implicit_recall": per_source.get("synthetic_implicit", {}).get("accuracy"),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dir(args.output_dir, args.overwrite)

    base_model_path = resolve_checkpoint(args.base_model_path)
    print(f"Base checkpoint: {base_model_path}")
    print(f"Output directory: {args.output_dir}")

    real_train = load_real_split(args.real_train_path, "train", args.limit_real_train_samples)
    real_val = load_real_split(args.real_val_path, "validation", args.limit_eval_samples)
    synthetic_pool = load_synthetic_train(args.synthetic_train_path, args.limit_synthetic_train_samples)
    selected_synthetic = select_synthetic_examples(
        synthetic_pool,
        real_n=len(real_train),
        ratio=args.synthetic_ratio,
        sampling=args.synthetic_sampling,
        seed=args.seed,
    )
    mixed_train = build_mixed_train(real_train, selected_synthetic, args.seed)

    real_world_test = load_real_world_tests(args)
    synthetic_benchmark = load_synthetic_benchmarks(args)
    assert_safety(real_val, real_world_test, synthetic_pool, synthetic_benchmark)

    training_summary = {
        "real_train_examples": int(len(real_train)),
        "synthetic_pool_examples": int(len(synthetic_pool)),
        "selected_synthetic_examples": int(len(selected_synthetic)),
        "mixed_train_examples": int(len(mixed_train)),
        "synthetic_share": float(mixed_train["origin"].eq("synthetic").mean()),
        "real_validation_examples": int(len(real_val)),
        "real_train_class_counts": class_counts(real_train),
        "selected_synthetic_class_counts": class_counts(selected_synthetic),
        "mixed_train_class_counts": class_counts(mixed_train),
        "real_validation_class_counts": class_counts(real_val),
    }
    (args.output_dir / "training_data_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    config = vars(args).copy()
    config["base_model_path"] = str(base_model_path)
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    print(f"Device: {device}")
    print(f"Using fp16: {use_fp16}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=2)
    model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=mixed_train["binary_label"].to_numpy(),
    )
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    eval_loss_fn = torch.nn.CrossEntropyLoss()
    print(f"Training class weights: {class_weights}")

    train_loader = make_loader(mixed_train, tokenizer, args.max_length, args.batch_size, True, args.num_workers)
    val_loader = make_loader(real_val, tokenizer, args.max_length, args.eval_batch_size, False, args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * args.warmup_ratio), 1),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16) if use_fp16 else None

    best_f1 = -1.0
    best_epoch = 0
    best_dir = args.output_dir / "best_model"
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_f1, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn, scaler, use_fp16
        )
        val_loss, val_labels, val_probs = predict_proba(model, val_loader, device, eval_loss_fn, "Real validation")
        val_preds = preds_at_threshold(val_probs, 0.5)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Saved new best checkpoint to {best_dir}")
        print(f"Train loss: {train_loss:.4f} | macro F1: {train_f1:.4f} | acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f} | macro F1: {val_f1:.4f} | acc: {val_acc:.4f}")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "is_best": improved,
            }
        )

    final_dir = args.output_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nReloading best checkpoint from epoch {best_epoch}: {best_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(best_dir, num_labels=2)
    model.to(device)
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "history": history}, indent=2),
        encoding="utf-8",
    )

    val_loss, val_labels, val_probs = predict_proba(model, val_loader, device, eval_loss_fn, "Real validation threshold")
    tuned_threshold = None
    if args.tune_threshold:
        threshold_payload = tune_threshold(val_labels, val_probs)
        threshold_payload["validation_loss"] = val_loss
        (args.output_dir / "threshold.json").write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")
        tuned_threshold = float(threshold_payload["threshold"])
        print(f"Tuned threshold: {tuned_threshold:.2f} | val macro F1={threshold_payload['macro_f1']:.4f}")

    print(
        f"\nReal-world test: {len(real_world_test):,} rows | "
        f"sources={real_world_test['source'].value_counts().sort_index().to_dict()}"
    )
    real_loader = make_loader(real_world_test, tokenizer, args.max_length, args.eval_batch_size, False, args.num_workers)
    _, real_labels, real_probs = predict_proba(model, real_loader, device, eval_loss_fn, "Real-world test")
    real_05 = save_eval_outputs("real_world", args.output_dir, real_world_test, real_labels, real_probs, 0.5)

    print(
        f"\nSynthetic benchmark: {len(synthetic_benchmark):,} rows | "
        f"sources={synthetic_benchmark['source'].value_counts().sort_index().to_dict()}"
    )
    syn_loader = make_loader(synthetic_benchmark, tokenizer, args.max_length, args.eval_batch_size, False, args.num_workers)
    _, syn_labels, syn_probs = predict_proba(model, syn_loader, device, eval_loss_fn, "Synthetic benchmark")
    syn_05 = save_eval_outputs("synthetic_benchmarks", args.output_dir, synthetic_benchmark, syn_labels, syn_probs, 0.5)
    syn_diag_05 = synthetic_diagnostics(synthetic_benchmark, syn_probs, 0.5)
    (args.output_dir / "synthetic_diagnostic_metrics_threshold_0p5.json").write_text(
        json.dumps(syn_diag_05, indent=2),
        encoding="utf-8",
    )

    if tuned_threshold is not None:
        real_tuned = save_eval_outputs(
            "real_world", args.output_dir, real_world_test, real_labels, real_probs, tuned_threshold
        )
        syn_tuned = save_eval_outputs(
            "synthetic_benchmarks", args.output_dir, synthetic_benchmark, syn_labels, syn_probs, tuned_threshold
        )
        syn_diag_tuned = synthetic_diagnostics(synthetic_benchmark, syn_probs, tuned_threshold)
        (args.output_dir / "synthetic_diagnostic_metrics_threshold_tuned.json").write_text(
            json.dumps(syn_diag_tuned, indent=2),
            encoding="utf-8",
        )
        write_model_comparison(args.output_dir, real_tuned, syn_tuned)
    else:
        write_model_comparison(args.output_dir, real_05, syn_05)

    print(
        "Real-world threshold 0.5: "
        f"accuracy={real_05['accuracy']:.4f} | macro_f1={real_05['macro_f1']:.4f} | "
        f"positive_rate={real_05['positive_prediction_rate']:.4f}"
    )
    print(
        "Synthetic threshold 0.5: "
        f"accuracy={syn_05['accuracy']:.4f} | macro_f1={syn_05['macro_f1']:.4f}"
    )
    if tuned_threshold is not None:
        print(f"Tuned-threshold outputs saved with threshold={tuned_threshold:.2f}")
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
