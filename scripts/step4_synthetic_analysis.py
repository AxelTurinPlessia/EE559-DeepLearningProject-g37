"""Evaluate a fine-tuned RoBERTa checkpoint on synthetic diagnostics.

Example:
    python scripts/step4_synthetic_analysis.py \
      --model_path results/roberta_mixed_post_ocr/lr2e-5_seed42 \
      --emoji_pairs_path datasets/synthetic/pairs.jsonl \
      --implicit_path datasets/synthetic/implicit.jsonl \
      --neutral_path datasets/synthetic/counterexamples.jsonl \
      --explicit_path datasets/synthetic/explicit.json \
      --output_dir results/synthetic_diagnostics \
      --batch_size 32 \
      --max_length 256 \
      --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import emoji
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_PATH = Path("results/roberta_mixed_post_ocr/lr2e-5_seed42")
DEFAULT_OUTPUT_DIR = Path("results/synthetic_diagnostics")
LABEL_TEXT = {0: "non_misogynistic", 1: "misogynistic"}
MISSING_TEXT = "unknown"
CORE_COLUMNS = [
    "id",
    "text",
    "label",
    "y_true",
    "p_misogynistic",
    "y_pred",
    "source_dataset",
    "sexism_type",
    "category",
    "contains_emoji",
    "emoji_role",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic misogyny diagnostics with a fine-tuned RoBERTa checkpoint."
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--emoji_pairs_path", type=Path, default=Path("datasets/synthetic/pairs.jsonl"))
    parser.add_argument("--implicit_path", type=Path, default=Path("datasets/synthetic/implicit.jsonl"))
    parser.add_argument("--neutral_path", type=Path, default=Path("datasets/synthetic/counterexamples.jsonl"))
    parser.add_argument("--explicit_path", type=Path, default=Path("datasets/synthetic/explicit.json"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def resolve_model_path(model_path: Path) -> Path:
    if (model_path / "config.json").exists():
        return model_path
    best_model = model_path / "best_model"
    if (best_model / "config.json").exists():
        print(f"Model path points to a run directory; using checkpoint: {best_model}")
        return best_model
    raise FileNotFoundError(
        f"Missing model checkpoint config.json at {model_path} or {best_model}"
    )


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    require_path(path, "dataset")
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(obj)
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "examples", "rows", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [data]
    raise ValueError(f"Unsupported JSON structure in {path}")


def normalize_label(value: Any, source_dataset: str | None = None) -> int:
    if value is None or value == "":
        if source_dataset in {"implicit", "explicit"}:
            return 1
        if source_dataset in {"neutral", "counterexamples"}:
            return 0
        raise ValueError("Missing label and cannot infer it from source_dataset")

    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return int(value)
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)

    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = text.replace("__", "_")
    if text in {"0", "false", "non_misogynistic", "not_misogynistic", "nonmisogynistic"}:
        return 0
    if text in {"not_sexist", "non_sexist", "none", "neutral"}:
        return 0
    if text in {"1", "true", "misogynistic", "sexist"}:
        return 1
    if "non" in text or text.startswith("not_"):
        return 0
    if "misogyn" in text or "sexist" in text:
        return 1
    raise ValueError(f"Cannot normalize label: {value!r}")


def has_emoji(text: Any) -> bool:
    return any(char in emoji.EMOJI_DATA for char in str(text))


def stringify_metadata(value: Any) -> str:
    if value is None or value == "":
        return MISSING_TEXT
    if isinstance(value, list):
        return "|".join(str(item) for item in value) if value else MISSING_TEXT
    return str(value)


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def first_existing(row: dict[str, Any], candidates: Iterable[str], default: Any = "") -> Any:
    for key in candidates:
        if key in row and not is_missing(row[key]):
            return row[key]
    return default


def standardize_examples(rows: list[dict[str, Any]], source_dataset: str) -> pd.DataFrame:
    standardized = []
    for idx, row in enumerate(rows, start=1):
        text = first_existing(row, ("text", "post_text", "sentence", "content"), "")
        label = first_existing(row, ("label", "gold_label", "target", "y_true"), "")
        y_true = normalize_label(label, source_dataset=source_dataset)
        contains = row.get("contains_emoji", has_emoji(text))
        standardized.append(
            {
                "id": stringify_metadata(first_existing(row, ("id", "sample_id"), f"{source_dataset}_{idx:04d}")),
                "text": str(text),
                "label": stringify_metadata(label if label != "" else LABEL_TEXT[y_true]),
                "y_true": y_true,
                "source_dataset": source_dataset,
                "sexism_type": stringify_metadata(row.get("sexism_type", source_dataset)),
                "category": stringify_metadata(row.get("category", MISSING_TEXT)),
                "contains_emoji": bool(contains),
                "emoji_role": stringify_metadata(row.get("emoji_role", MISSING_TEXT)),
                "rationale": stringify_metadata(row.get("rationale", MISSING_TEXT)),
                "gender_terms": stringify_metadata(row.get("gender_terms", MISSING_TEXT)),
            }
        )
    return pd.DataFrame(standardized)


class TextDataset(Dataset):
    def __init__(self, texts: Iterable[str], tokenizer, max_length: int) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoding.items()}


def find_positive_class_index(model) -> int:
    config = model.config
    candidates: list[tuple[int, str]] = []
    id2label = getattr(config, "id2label", None) or {}
    for raw_idx, label in id2label.items():
        label_text = str(label).lower().replace("-", "_").replace(" ", "_")
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        candidates.append((idx, label_text))
    for idx, label in candidates:
        if "misogyn" in label or label in {"sexist", "positive"}:
            return idx
    for idx, label in candidates:
        if label == "label_1":
            print(
                "WARNING: model config uses generic LABEL_0/LABEL_1 names; "
                "assuming class index 1 is misogynistic."
            )
            return idx

    label2id = getattr(config, "label2id", None) or {}
    for raw_label, raw_idx in label2id.items():
        label = str(raw_label).lower().replace("-", "_").replace(" ", "_")
        if "misogyn" in label or label in {"sexist", "positive"}:
            return int(raw_idx)
    for raw_label, raw_idx in label2id.items():
        label = str(raw_label).lower().replace("-", "_").replace(" ", "_")
        if label == "label_1":
            print(
                "WARNING: model config uses generic LABEL_0/LABEL_1 names; "
                "assuming class index 1 is misogynistic."
            )
            return int(raw_idx)

    print("WARNING: model config has no usable id2label/label2id; assuming class index 1 is misogynistic.")
    return 1


def predict_texts(
    texts: Iterable[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    threshold: float,
    positive_class_index: int,
) -> tuple[list[float], list[int]]:
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities: list[float] = []
    predictions: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, positive_class_index]
            probabilities.extend(probs.detach().cpu().tolist())
            predictions.extend((probs >= threshold).long().detach().cpu().tolist())
    return probabilities, predictions


def add_predictions(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    args: argparse.Namespace,
    positive_class_index: int,
) -> pd.DataFrame:
    if df.empty:
        return df.assign(p_misogynistic=[], y_pred=[])
    probs, preds = predict_texts(
        df["text"].astype(str).tolist(),
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threshold=args.threshold,
        positive_class_index=positive_class_index,
    )
    out = df.copy()
    out["p_misogynistic"] = probs
    out["y_pred"] = preds
    return out


def safe_metric(func, y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(func(y_true, y_pred))


def recall_for_positive(df: pd.DataFrame) -> float:
    positives = df[df["y_true"].eq(1)]
    if positives.empty:
        return float("nan")
    return float(positives["y_pred"].eq(1).mean())


def false_positive_rate(df: pd.DataFrame) -> float:
    negatives = df[df["y_true"].eq(0)]
    if negatives.empty:
        return float("nan")
    return float(negatives["y_pred"].eq(1).mean())


def category_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["source_dataset", "sexism_type", "category"], dropna=False):
        source, sexism_type, category = keys
        rows.append(
            {
                "source_dataset": source,
                "sexism_type": sexism_type,
                "category": category,
                "n": len(group),
                "positive_count": int(group["y_true"].sum()),
                "negative_count": int(group["y_true"].eq(0).sum()),
                "accuracy": safe_metric(accuracy_score, group["y_true"], group["y_pred"]),
                "recall": recall_for_positive(group),
                "false_positive_rate": false_positive_rate(group),
                "avg_p_misogynistic": float(group["p_misogynistic"].mean()),
            }
        )
    return pd.DataFrame(rows)


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    return pd.DataFrame(
        [
            {
                "n": len(df),
                "accuracy": accuracy_score(df["y_true"], df["y_pred"]),
                "precision": precision_score(df["y_true"], df["y_pred"], zero_division=0),
                "recall": recall_score(df["y_true"], df["y_pred"], zero_division=0),
                "macro_f1": f1_score(df["y_true"], df["y_pred"], average="macro", zero_division=0),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        ]
    )


def explicit_vs_implicit_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in ("explicit", "implicit"):
        subset = df[df["source_dataset"].eq(source)]
        rows.append(
            {
                "source_dataset": source,
                "n": len(subset),
                "recall": recall_for_positive(subset),
                "avg_p_misogynistic": float(subset["p_misogynistic"].mean()),
            }
        )
    explicit = rows[0]
    implicit = rows[1]
    rows.append(
        {
            "source_dataset": "explicit_minus_implicit",
            "n": "",
            "recall": explicit["recall"] - implicit["recall"],
            "avg_p_misogynistic": explicit["avg_p_misogynistic"] - implicit["avg_p_misogynistic"],
        }
    )
    return pd.DataFrame(rows)


def neutral_summary(df: pd.DataFrame) -> pd.DataFrame:
    neutral = df[df["source_dataset"].eq("neutral")]
    rows = [
        {
            "group": "all_neutral_counterexamples",
            "n": len(neutral),
            "false_positive_rate": false_positive_rate(neutral),
            "avg_p_misogynistic": float(neutral["p_misogynistic"].mean()) if len(neutral) else float("nan"),
        }
    ]
    if "gender_terms" in neutral.columns:
        exploded = neutral.copy()
        exploded["gender_term"] = exploded["gender_terms"].apply(
            lambda value: [part for part in str(value).split("|") if part and part != MISSING_TEXT]
        )
        exploded = exploded.explode("gender_term")
        for term, group in exploded.dropna(subset=["gender_term"]).groupby("gender_term"):
            rows.append(
                {
                    "group": f"gender_term:{term}",
                    "n": len(group),
                    "false_positive_rate": false_positive_rate(group),
                    "avg_p_misogynistic": float(group["p_misogynistic"].mean()),
                }
            )
    return pd.DataFrame(rows)


def standardize_pairs(rows: list[dict[str, Any]]) -> pd.DataFrame:
    long_rows = []
    for idx, row in enumerate(rows, start=1):
        base_pair_id = stringify_metadata(first_existing(row, ("base_pair_id", "pair_id", "id"), f"pair_{idx:04d}"))
        label = first_existing(row, ("label", "gold_label", "target"), "")
        y_true = normalize_label(label, source_dataset=None)
        common = {
            "base_pair_id": base_pair_id,
            "label": stringify_metadata(label if label != "" else LABEL_TEXT[y_true]),
            "y_true": y_true,
            "sexism_type": stringify_metadata(row.get("sexism_type", MISSING_TEXT)),
            "category": stringify_metadata(row.get("category", MISSING_TEXT)),
            "emoji_role": stringify_metadata(row.get("emoji_role", MISSING_TEXT)),
            "targeted_failure_mode": stringify_metadata(row.get("targeted_failure_mode", MISSING_TEXT)),
            "rationale": stringify_metadata(row.get("rationale", MISSING_TEXT)),
        }
        text_map = {
            "plain": first_existing(row, ("plain_text", "text_plain", "base_text", "plain"), ""),
            "emoji": first_existing(row, ("emoji_text", "text_with_emoji", "emoji"), ""),
        }
        for condition, text in text_map.items():
            long_rows.append({**common, "condition": condition, "text": str(text)})
    return pd.DataFrame(long_rows)


def build_pair_level(pair_long: pd.DataFrame) -> pd.DataFrame:
    plain = pair_long[pair_long["condition"].eq("plain")].set_index("base_pair_id")
    emoji_df = pair_long[pair_long["condition"].eq("emoji")].set_index("base_pair_id")
    common_columns = [
        "label",
        "y_true",
        "sexism_type",
        "category",
        "emoji_role",
        "targeted_failure_mode",
        "rationale",
    ]
    pair = plain[common_columns].copy()
    pair["plain_text"] = plain["text"]
    pair["emoji_text"] = emoji_df["text"]
    pair["p_misogynistic_plain"] = plain["p_misogynistic"]
    pair["p_misogynistic_emoji"] = emoji_df["p_misogynistic"]
    pair["y_pred_plain"] = plain["y_pred"]
    pair["y_pred_emoji"] = emoji_df["y_pred"]
    pair["delta_p"] = pair["p_misogynistic_emoji"] - pair["p_misogynistic_plain"]
    pair["flip"] = pair["y_pred_plain"] != pair["y_pred_emoji"]
    pair["plain_correct"] = pair["y_pred_plain"] == pair["y_true"]
    pair["emoji_correct"] = pair["y_pred_emoji"] == pair["y_true"]

    def flip_type(row: pd.Series) -> str:
        if not row["flip"]:
            return "no_flip"
        if not row["plain_correct"] and row["emoji_correct"]:
            return "helpful_flip"
        if row["plain_correct"] and not row["emoji_correct"]:
            return "harmful_flip"
        if row["plain_correct"] and row["emoji_correct"]:
            return "correct_to_correct_flip"
        return "wrong_to_wrong_flip"

    pair["flip_type"] = pair.apply(flip_type, axis=1)
    return pair.reset_index()


def binary_recall(preds: pd.Series, truth: pd.Series) -> float:
    mask = truth.eq(1)
    if not mask.any():
        return float("nan")
    return float(preds[mask].eq(1).mean())


def binary_fpr(preds: pd.Series, truth: pd.Series) -> float:
    mask = truth.eq(0)
    if not mask.any():
        return float("nan")
    return float(preds[mask].eq(1).mean())


def emoji_summary(pair: pd.DataFrame) -> pd.DataFrame:
    flip_counts = pair["flip_type"].value_counts().to_dict()
    row = {
        "n_pairs": len(pair),
        "flip_rate": float(pair["flip"].mean()),
        "mean_delta_p": float(pair["delta_p"].mean()),
        "median_delta_p": float(pair["delta_p"].median()),
        "mean_abs_delta_p": float(pair["delta_p"].abs().mean()),
        "plain_accuracy": float(pair["plain_correct"].mean()),
        "emoji_accuracy": float(pair["emoji_correct"].mean()),
        "plain_recall_misogynistic": binary_recall(pair["y_pred_plain"], pair["y_true"]),
        "emoji_recall_misogynistic": binary_recall(pair["y_pred_emoji"], pair["y_true"]),
        "plain_fpr_non_misogynistic": binary_fpr(pair["y_pred_plain"], pair["y_true"]),
        "emoji_fpr_non_misogynistic": binary_fpr(pair["y_pred_emoji"], pair["y_true"]),
    }
    for name in [
        "no_flip",
        "helpful_flip",
        "harmful_flip",
        "wrong_to_wrong_flip",
        "correct_to_correct_flip",
    ]:
        row[f"{name}_count"] = int(flip_counts.get(name, 0))
        row[f"{name}_proportion"] = float(flip_counts.get(name, 0) / len(pair)) if len(pair) else 0.0
    return pd.DataFrame([row])


def emoji_group_summary(pair: pd.DataFrame, group_column: str, include_helpful: bool = True) -> pd.DataFrame:
    rows = []
    for value, group in pair.groupby(group_column, dropna=False):
        row = {
            group_column: value,
            "n": len(group),
            "flip_rate": float(group["flip"].mean()),
            "mean_delta_p": float(group["delta_p"].mean()),
            "mean_abs_delta_p": float(group["delta_p"].abs().mean()),
        }
        if include_helpful:
            row["harmful_flip_rate"] = float(group["flip_type"].eq("harmful_flip").mean())
            row["helpful_flip_rate"] = float(group["flip_type"].eq("helpful_flip").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = list(view.columns)
    rows = [[str(row[column]) for column in headers] for _, row in view.iterrows()]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows)) if rows else len(str(header))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def generate_report(
    output_dir: Path,
    model_path: Path,
    dataset_paths: dict[str, Path],
    threshold: float,
    all_predictions: pd.DataFrame,
    overall: pd.DataFrame,
    explicit_vs_implicit: pd.DataFrame,
    neutral_metrics: pd.DataFrame,
    category: pd.DataFrame,
    emoji_pair_summary: pd.DataFrame,
    emoji_by_role: pd.DataFrame,
    emoji_by_category: pd.DataFrame,
    worst_implicit_fn: pd.DataFrame,
    worst_neutral_fp: pd.DataFrame,
    harmful_flips: pd.DataFrame,
) -> None:
    counts = all_predictions["source_dataset"].value_counts().rename_axis("dataset").reset_index(name="n")
    evi = explicit_vs_implicit.set_index("source_dataset")
    explicit_recall = evi.loc["explicit", "recall"]
    implicit_recall = evi.loc["implicit", "recall"]
    recall_gap = evi.loc["explicit_minus_implicit", "recall"]
    neutral_fpr = neutral_metrics.iloc[0]["false_positive_rate"]
    emoji_flip_rate = emoji_pair_summary.iloc[0]["flip_rate"]
    emoji_mean_delta = emoji_pair_summary.iloc[0]["mean_delta_p"]

    implicit_problem_categories = (
        category[category["source_dataset"].eq("implicit")]
        .sort_values(["recall", "n"], ascending=[True, False])
        [["category", "n", "recall", "avg_p_misogynistic"]]
    )
    neutral_overflagged = (
        category[category["source_dataset"].eq("neutral")]
        .sort_values(["false_positive_rate", "n"], ascending=[False, False])
        [["category", "n", "false_positive_rate", "avg_p_misogynistic"]]
    )
    unstable_roles = emoji_by_role.sort_values(["flip_rate", "n"], ascending=[False, False])
    unstable_categories = emoji_by_category.sort_values(["flip_rate", "n"], ascending=[False, False])

    lines = [
        "# Synthetic Diagnostic Report",
        "",
        f"- Model path: `{model_path}`",
        f"- Threshold: `{threshold}`",
        "- Dataset paths:",
        *[f"  - {name}: `{path}`" for name, path in dataset_paths.items()],
        "",
        "## Dataset Counts",
        markdown_table(counts),
        "",
        "## Main Metrics",
        f"- Explicit recall: `{explicit_recall:.4f}`",
        f"- Implicit recall: `{implicit_recall:.4f}`",
        f"- Explicit-implicit recall gap: `{recall_gap:.4f}`",
        f"- Neutral false positive rate: `{neutral_fpr:.4f}`",
        f"- Emoji flip rate: `{emoji_flip_rate:.4f}`",
        f"- Mean emoji delta_p: `{emoji_mean_delta:.4f}`",
        "",
        "## Overall Synthetic Metrics",
        markdown_table(overall),
        "",
        "## Most Problematic Implicit Categories",
        markdown_table(implicit_problem_categories, max_rows=10),
        "",
        "## Most Overflagged Neutral Categories",
        markdown_table(neutral_overflagged, max_rows=10),
        "",
        "## Most Unstable Emoji Roles",
        markdown_table(unstable_roles, max_rows=10),
        "",
        "## Most Unstable Emoji Categories",
        markdown_table(unstable_categories, max_rows=10),
        "",
        "## Five Worst Implicit False Negatives",
        markdown_table(
            worst_implicit_fn[["text", "category", "p_misogynistic", "rationale"]],
            max_rows=5,
        ),
        "",
        "## Five Worst Neutral False Positives",
        markdown_table(
            worst_neutral_fp[["text", "category", "p_misogynistic", "gender_terms", "rationale"]],
            max_rows=5,
        ),
        "",
        "## Five Largest Harmful Emoji Flips",
        markdown_table(
            harmful_flips[
                [
                    "base_pair_id",
                    "plain_text",
                    "emoji_text",
                    "category",
                    "emoji_role",
                    "delta_p",
                    "flip_type",
                ]
            ],
            max_rows=5,
        ),
        "",
    ]
    (output_dir / "synthetic_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def print_main_metrics(
    overall: pd.DataFrame,
    explicit_vs_implicit: pd.DataFrame,
    neutral_metrics: pd.DataFrame,
    emoji_pair_summary: pd.DataFrame,
) -> None:
    overall_row = overall.iloc[0]
    evi = explicit_vs_implicit.set_index("source_dataset")
    neutral_row = neutral_metrics.iloc[0]
    emoji_row = emoji_pair_summary.iloc[0]
    print("\n=== Synthetic Diagnostic Summary ===")
    print(f"Examples: {int(overall_row['n'])}")
    print(f"Accuracy: {overall_row['accuracy']:.4f}")
    print(f"Precision: {overall_row['precision']:.4f}")
    print(f"Recall: {overall_row['recall']:.4f}")
    print(f"Macro F1: {overall_row['macro_f1']:.4f}")
    print(
        "Confusion matrix: "
        f"TN={int(overall_row['tn'])}, FP={int(overall_row['fp'])}, "
        f"FN={int(overall_row['fn'])}, TP={int(overall_row['tp'])}"
    )
    print("\nExplicit vs implicit:")
    print(f"  Explicit recall: {evi.loc['explicit', 'recall']:.4f}")
    print(f"  Implicit recall: {evi.loc['implicit', 'recall']:.4f}")
    print(f"  Recall gap: {evi.loc['explicit_minus_implicit', 'recall']:.4f}")
    print("\nNeutral/counterexamples:")
    print(f"  False positive rate: {neutral_row['false_positive_rate']:.4f}")
    print(f"  Avg p_misogynistic: {neutral_row['avg_p_misogynistic']:.4f}")
    print("\nEmoji pairs:")
    print(f"  Pairs: {int(emoji_row['n_pairs'])}")
    print(f"  Flip rate: {emoji_row['flip_rate']:.4f}")
    print(f"  Mean delta_p: {emoji_row['mean_delta_p']:.4f}")
    print(f"  Mean abs delta_p: {emoji_row['mean_abs_delta_p']:.4f}")
    print(f"  Plain accuracy: {emoji_row['plain_accuracy']:.4f}")
    print(f"  Emoji accuracy: {emoji_row['emoji_accuracy']:.4f}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model_path)
    for path, description in [
        (args.emoji_pairs_path, "emoji pairs dataset"),
        (args.implicit_path, "implicit dataset"),
        (args.neutral_path, "neutral/counterexamples dataset"),
        (args.explicit_path, "explicit dataset"),
    ]:
        require_path(path, description)

    print(f"Loading model: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    positive_class_index = find_positive_class_index(model)
    print(f"Positive class index: {positive_class_index}")

    implicit_df = standardize_examples(load_json_or_jsonl(args.implicit_path), "implicit")
    neutral_df = standardize_examples(load_json_or_jsonl(args.neutral_path), "neutral")
    explicit_df = standardize_examples(load_json_or_jsonl(args.explicit_path), "explicit")
    all_predictions = pd.concat([implicit_df, neutral_df, explicit_df], ignore_index=True)
    all_predictions = add_predictions(
        all_predictions,
        tokenizer,
        model,
        device,
        args,
        positive_class_index,
    )

    write_csv(all_predictions[CORE_COLUMNS + ["rationale", "gender_terms"]], args.output_dir / "synthetic_all_predictions.csv")

    overall = overall_summary(all_predictions)
    category = category_metrics(all_predictions)
    explicit_implicit = explicit_vs_implicit_summary(all_predictions)
    neutral_metrics = neutral_summary(all_predictions)
    write_csv(overall, args.output_dir / "synthetic_summary_overall.csv")
    write_csv(category, args.output_dir / "synthetic_category_metrics.csv")
    write_csv(explicit_implicit, args.output_dir / "synthetic_explicit_vs_implicit.csv")
    write_csv(neutral_metrics, args.output_dir / "synthetic_neutral_counterexample_metrics.csv")

    pair_long = standardize_pairs(load_json_or_jsonl(args.emoji_pairs_path))
    pair_long = add_predictions(pair_long, tokenizer, model, device, args, positive_class_index)
    pair_level = build_pair_level(pair_long)
    pair_summary = emoji_summary(pair_level)
    pair_by_role = emoji_group_summary(pair_level, "emoji_role")
    pair_by_category = emoji_group_summary(pair_level, "category")
    pair_by_sexism_type = emoji_group_summary(pair_level, "sexism_type", include_helpful=False)
    flip_examples = pair_level[pair_level["flip"]].copy()
    flip_examples = flip_examples.reindex(flip_examples["delta_p"].abs().sort_values(ascending=False).index)

    write_csv(pair_long, args.output_dir / "emoji_pairs_long_predictions.csv")
    write_csv(pair_level, args.output_dir / "emoji_pairs_pair_level.csv")
    write_csv(pair_summary, args.output_dir / "emoji_pairs_summary.csv")
    write_csv(pair_by_role, args.output_dir / "emoji_pairs_by_role.csv")
    write_csv(pair_by_category, args.output_dir / "emoji_pairs_by_category.csv")
    write_csv(pair_by_sexism_type, args.output_dir / "emoji_pairs_by_sexism_type.csv")
    write_csv(flip_examples, args.output_dir / "emoji_pairs_flip_examples.csv")

    worst_implicit_fn = all_predictions[
        all_predictions["source_dataset"].eq("implicit")
        & all_predictions["y_true"].eq(1)
        & all_predictions["y_pred"].eq(0)
    ].sort_values("p_misogynistic", ascending=True)
    worst_neutral_fp = all_predictions[
        all_predictions["source_dataset"].eq("neutral")
        & all_predictions["y_true"].eq(0)
        & all_predictions["y_pred"].eq(1)
    ].sort_values("p_misogynistic", ascending=False)
    harmful_flips = pair_level[pair_level["flip_type"].eq("harmful_flip")].copy()
    harmful_flips = harmful_flips.reindex(harmful_flips["delta_p"].abs().sort_values(ascending=False).index)

    write_csv(
        worst_implicit_fn[
            ["text", "category", "sexism_type", "p_misogynistic", "y_pred", "label", "rationale"]
        ],
        args.output_dir / "worst_false_negatives_implicit.csv",
    )
    write_csv(
        worst_neutral_fp[
            ["text", "category", "p_misogynistic", "y_pred", "label", "gender_terms", "rationale"]
        ],
        args.output_dir / "worst_false_positives_neutral.csv",
    )
    write_csv(
        harmful_flips[
            [
                "base_pair_id",
                "plain_text",
                "emoji_text",
                "label",
                "category",
                "emoji_role",
                "p_misogynistic_plain",
                "p_misogynistic_emoji",
                "delta_p",
                "y_pred_plain",
                "y_pred_emoji",
                "flip_type",
                "rationale",
            ]
        ],
        args.output_dir / "harmful_emoji_flips.csv",
    )

    generate_report(
        output_dir=args.output_dir,
        model_path=model_path,
        dataset_paths={
            "emoji_pairs": args.emoji_pairs_path,
            "implicit": args.implicit_path,
            "neutral": args.neutral_path,
            "explicit": args.explicit_path,
        },
        threshold=args.threshold,
        all_predictions=all_predictions,
        overall=overall,
        explicit_vs_implicit=explicit_implicit,
        neutral_metrics=neutral_metrics,
        category=category,
        emoji_pair_summary=pair_summary,
        emoji_by_role=pair_by_role,
        emoji_by_category=pair_by_category,
        worst_implicit_fn=worst_implicit_fn,
        worst_neutral_fp=worst_neutral_fp,
        harmful_flips=harmful_flips,
    )

    print_main_metrics(overall, explicit_implicit, neutral_metrics, pair_summary)
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
