"""Show the final project results without retraining any model.

The script reads stored metrics for the text-only RoBERTa experiments and can
evaluate the saved MAMI CLIP/RoBERTa+CLIP checkpoints on the validation split.
MAMI metrics are cached after evaluation so a screencast can rerun quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABEL_NAMES = ["non-misogynistic", "misogynistic"]


@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
    kind: str


@dataclass(frozen=True)
class MamiModelSpec:
    key: str
    name: str
    model_type: str
    weights_path: Path
    cache_path: Path
    text_checkpoint: Path | None = None


TEXT_ARTIFACTS = [
    Artifact("RoBERTa EDOS", Path("results/roberta_base/best_model"), "HF checkpoint"),
    Artifact("RoBERTa EDOS + Hatemoji", Path("results/roberta_emoji/model"), "HF checkpoint"),
    Artifact("RoBERTa mixed post/OCR", Path("results/roberta_mixed_post_ocr/best_model"), "HF checkpoint"),
    Artifact("RoBERTa mixed real + synthetic", Path("results/roberta_mixed_synthetic/best_model"), "HF checkpoint"),
]

MAMI_MODELS = [
    MamiModelSpec(
        key="clip_only",
        name="CLIP only",
        model_type="clip",
        weights_path=Path("results/results_clip_mami/model.pt"),
        cache_path=Path("results/results_clip_mami/metrics_validation.json"),
    ),
    MamiModelSpec(
        key="roberta_base_clip",
        name="RoBERTa EDOS + CLIP",
        model_type="multimodal",
        weights_path=Path("results/results_multimodal_roberta_base/model.pt"),
        cache_path=Path("results/results_multimodal_roberta_base/metrics_validation.json"),
        text_checkpoint=Path("results/roberta_base/best_model"),
    ),
    MamiModelSpec(
        key="roberta_synthetic_clip",
        name="RoBERTa synthetic + CLIP",
        model_type="multimodal",
        weights_path=Path("results/results_multimodal_roberta_synthetic/model.pt"),
        cache_path=Path("results/results_multimodal_roberta_synthetic/metrics_validation.json"),
        text_checkpoint=Path("results/roberta_mixed_synthetic/best_model"),
    ),
    MamiModelSpec(
        key="roberta_mixed_clip",
        name="RoBERTa mixed + CLIP",
        model_type="multimodal",
        weights_path=Path("results/results_multimodal_weighted/model.pt"),
        cache_path=Path("results/results_multimodal_weighted/metrics_validation.json"),
        text_checkpoint=Path("results/roberta_mixed_post_ocr/best_model"),
    ),
]

MAMI_SPLIT_LABELS = {
    "validation": "held-out eval",
    "test": "test",
}

REQUIRED_RESULT_PATHS = [
    Path("results/roberta_base/metrics.json"),
    Path("results/roberta_base/post_ocr_predictions.csv"),
    Path("results/roberta_base/best_model/model.safetensors"),
    Path("results/roberta_emoji/metrics.json"),
    Path("results/roberta_emoji/post_ocr_predictions.csv"),
    Path("results/roberta_emoji/model/model.safetensors"),
    Path("results/roberta_mixed_post_ocr/metrics.json"),
    Path("results/roberta_mixed_post_ocr/test_predictions.csv"),
    Path("results/roberta_mixed_post_ocr/best_model/model.safetensors"),
    Path("results/roberta_mixed_synthetic/metrics_real_world_threshold_0p5.json"),
    Path("results/roberta_mixed_synthetic/metrics_synthetic_benchmarks_threshold_0p5.json"),
    Path("results/roberta_mixed_synthetic/synthetic_diagnostic_metrics_threshold_0p5.json"),
    Path("results/roberta_mixed_synthetic/best_model/model.safetensors"),
    Path("results/results_clip_mami/model.pt"),
    Path("results/results_multimodal_roberta_base/model.pt"),
    Path("results/results_multimodal_roberta_synthetic/model.pt"),
    Path("results/results_multimodal_weighted/model.pt"),
    Path("results/analysis_step4_roberta_mixed_post_ocr/edos_implicit_explicit_summary.csv"),
    Path("results/analysis_step4_roberta_mixed_post_ocr/gender_bias_analysis.csv"),
    Path("results/analysis_step4_roberta_mixed_post_ocr/full_predictions_with_analysis.csv"),
    Path("results/synthetic_diagnostics/synthetic_summary_overall.csv"),
    Path("results/synthetic_diagnostics/synthetic_explicit_vs_implicit.csv"),
    Path("results/synthetic_diagnostics/synthetic_neutral_counterexample_metrics.csv"),
    Path("results/synthetic_diagnostics/emoji_pairs_summary.csv"),
]

DATASET_PATHS = [
    Path("datasets/synthetic/synthetic_train_set.jsonl"),
    Path("datasets/synthetic/implicit.jsonl"),
    Path("datasets/synthetic/explicit.json"),
    Path("datasets/synthetic/counterexamples.jsonl"),
    Path("datasets/synthetic/pairs.jsonl"),
    Path("datasets/edos/data/edos_labelled_aggregated.csv"),
    Path("datasets/Hatemoji/HatemojiBuild/train.csv"),
    Path("datasets/Hatemoji/HatemojiBuild/validation.csv"),
    Path("datasets/Hatemoji/HatemojiBuild/test.csv"),
    Path("datasets/online-misogyny-eacl2021-main/data/final_labels.csv"),
    Path("datasets/post_ocr_dataset.csv"),
    Path("datasets/roberta_mixed_post_ocr/train.csv"),
    Path("datasets/roberta_mixed_post_ocr/val.csv"),
    Path("datasets/roberta_mixed_post_ocr/test.csv"),
    Path("datasets/MAMI/train.tsv"),
    Path("datasets/MAMI/validation.tsv"),
    Path("datasets/MAMI/MAMI_2022_images/training_images"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display final EE559 project results without model training."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--mami-split",
        choices=["validation", "test"],
        default="validation",
        help="MAMI split used when cached MAMI metrics must be recomputed.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for optional saved-checkpoint MAMI evaluation.",
    )
    parser.add_argument(
        "--limit-mami-samples",
        type=int,
        default=None,
        help="Evaluate only the first N MAMI rows for a quick smoke test.",
    )
    parser.add_argument(
        "--skip-mami-eval",
        action="store_true",
        help="Only print cached MAMI metrics. Do not load .pt checkpoints.",
    )
    parser.add_argument(
        "--refresh-mami-metrics",
        action="store_true",
        help="Recompute MAMI metrics even if cached metrics JSON files exist.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    missing_results = [
        resolve(args.results_dir, path)
        for path in REQUIRED_RESULT_PATHS
        if not resolve(args.results_dir, path).exists()
    ]
    if missing_results:
        print("ERROR: Missing required result artifacts for main.py:")
        for path in missing_results:
            print(f"  - {path}")
        print("\nRestore the selected results folders or rerun the corresponding scripts before running main.py.")
        raise SystemExit(1)

    missing_datasets = [
        args.data_dir / Path(*path.parts[1:])
        for path in DATASET_PATHS
        if not (args.data_dir / Path(*path.parts[1:])).exists()
    ]
    if missing_datasets:
        level = "ERROR" if args.refresh_mami_metrics else "WARNING"
        print(f"{level}: Some datasets are missing from {args.data_dir}:")
        for path in missing_datasets:
            print(f"  - {path}")
        if args.refresh_mami_metrics:
            print("\nMAMI data is required to recompute MAMI metrics with --refresh-mami-metrics.")
            raise SystemExit(1)
        print("Saved result summaries can still be displayed, but recomputation will require these datasets.\n")


def resolve(base: Path, path: Path) -> Path:
    if path.parts and path.parts[0] == "results":
        return base / Path(*path.parts[1:])
    return path


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_csv_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rendered)) if rendered else len(headers[i])
        for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rendered:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def summarize_artifacts(results_dir: Path) -> None:
    print_section("Checkpoint Inventory")
    rows: list[list[Any]] = []
    for artifact in TEXT_ARTIFACTS:
        path = resolve(results_dir, artifact.path)
        if artifact.kind == "HF checkpoint":
            weight_file = path / "model.safetensors"
            config_file = path / "config.json"
            status = "ready" if weight_file.exists() and config_file.exists() else "missing"
            size = file_size(weight_file)
        else:
            status = "ready" if path.exists() else "missing"
            size = file_size(path)
        rows.append([artifact.name, artifact.kind, str(path), status, size])

    for spec in MAMI_MODELS:
        path = resolve(results_dir, spec.weights_path)
        status = "ready" if path.exists() else "missing"
        rows.append([spec.name, "PyTorch checkpoint", str(path), status, file_size(path)])

    print_table(["Model", "Type", "Path", "Status", "Size"], rows)


def summarize_roberta(results_dir: Path) -> None:
    print_section("Text Model Results")
    rows: list[list[Any]] = []

    base = read_json(results_dir / "roberta_base/metrics.json") or {}
    base_predictions = metrics_from_prediction_csv(results_dir / "roberta_base/post_ocr_predictions.csv")
    rows.append(
        [
            "RoBERTa EDOS",
            "post/OCR",
            base_predictions.get("n_examples", "6356"),
            format_float(base.get("test_macro_f1")),
            format_float(base_predictions.get("accuracy")),
            format_float(base.get("test_loss")),
        ]
    )

    emoji = read_json(results_dir / "roberta_emoji/metrics.json") or {}
    emoji_predictions = metrics_from_prediction_csv(results_dir / "roberta_emoji/post_ocr_predictions.csv")
    rows.append(
        [
            "RoBERTa EDOS + Hatemoji",
            "post/OCR",
            emoji_predictions.get("n_examples", "6356"),
            format_float(emoji.get("test_macro_f1")),
            format_float(emoji_predictions.get("accuracy")),
            format_float(emoji.get("test_loss")),
        ]
    )

    mixed = read_json(results_dir / "roberta_mixed_post_ocr/metrics.json") or {}
    mixed_test = mixed.get("test", {})
    rows.append(
        [
            "RoBERTa mixed post/OCR",
            "held-out post/OCR",
            "954",
            format_float(mixed_test.get("macro_f1")),
            format_float(mixed_test.get("accuracy")),
            format_float(mixed_test.get("loss")),
        ]
    )

    synthetic = read_json(results_dir / "roberta_mixed_synthetic/metrics_real_world_threshold_0p5.json") or {}
    rows.append(
        [
            "RoBERTa mixed real + synthetic",
            "real-world combined",
            synthetic.get("n_examples", "n/a"),
            format_float(synthetic.get("macro_f1")),
            format_float(synthetic.get("accuracy")),
            "n/a",
        ]
    )

    synthetic_bench = read_json(results_dir / "roberta_mixed_synthetic/metrics_synthetic_benchmarks_threshold_0p5.json") or {}
    rows.append(
        [
            "RoBERTa mixed real + synthetic",
            "synthetic benchmark",
            synthetic_bench.get("n_examples", "n/a"),
            format_float(synthetic_bench.get("macro_f1")),
            format_float(synthetic_bench.get("accuracy")),
            "n/a",
        ]
    )

    print_table(["Model", "Evaluation", "N", "Macro F1", "Accuracy", "Loss"], rows)

    per_source = synthetic.get("per_source") or {}
    if per_source:
        print("\nRoBERTa mixed real + synthetic by real-world source:")
        source_rows = [
            [
                source,
                metrics.get("n", "n/a"),
                format_float(metrics.get("macro_f1")),
                format_float(metrics.get("accuracy")),
                format_float(metrics.get("positive_prediction_rate")),
            ]
            for source, metrics in sorted(per_source.items())
        ]
        print_table(["Source", "N", "Macro F1", "Accuracy", "Positive Rate"], source_rows)

    training_summary = read_json(results_dir / "roberta_mixed_synthetic/training_data_summary.json") or {}
    if training_summary:
        selected = training_summary.get("selected_synthetic_examples", "n/a")
        pool = training_summary.get("synthetic_pool_examples", "n/a")
        print("\nSynthetic augmentation setup:")
        print(
            "  mixed_train_examples={mixed} | real_train={real} | selected_synthetic={synthetic} | synthetic_pool={pool} | synthetic_share={share}".format(
                mixed=training_summary.get("mixed_train_examples", "n/a"),
                real=training_summary.get("real_train_examples", "n/a"),
                synthetic=selected,
                pool=pool,
                share=format_float(training_summary.get("synthetic_share")),
            )
        )
        if isinstance(selected, int) and isinstance(pool, int) and selected > pool:
            print("  note: selected_synthetic exceeds the pool, so this stored run reused synthetic examples.")


def metrics_from_prediction_csv(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    labels: list[int] = []
    preds: list[int] = []
    for row in rows:
        try:
            labels.append(int(row["true_label"]))
            preds.append(int(row["predicted_label"]))
        except (KeyError, TypeError, ValueError):
            return {}
    return binary_metrics(labels, preds)


def summarize_error_and_diagnostics(results_dir: Path) -> None:
    print_section("Analysis Highlights")

    print_edos_implicit_explicit(results_dir)
    print_gender_bias(results_dir)
    print_synthetic_diagnostic_comparison(results_dir)


def print_edos_implicit_explicit(results_dir: Path) -> None:
    summary = read_csv_rows(results_dir / "analysis_step4_roberta_mixed_post_ocr/edos_implicit_explicit_summary.csv")
    if not summary:
        print("EDOS implicit/explicit analysis: missing saved summary.")
        return

    rows = [
        [
            row.get("sexism_type", "n/a"),
            row.get("samples", "n/a"),
            format_float(row.get("recall")),
            format_float(row.get("macro_f1_on_nonsexist_plus_type")),
        ]
        for row in summary
        if row.get("sexism_type") in {"explicit", "implicit"}
    ]
    print("RoBERTa mixed post/OCR on EDOS by sexism type:")
    print_table(["Sexism Type", "Samples", "Recall", "Macro F1"], rows)


def print_gender_bias(results_dir: Path) -> None:
    bias_rows = read_csv_rows(results_dir / "analysis_step4_roberta_mixed_post_ocr/gender_bias_analysis.csv")
    prediction_rows = read_csv_rows(results_dir / "analysis_step4_roberta_mixed_post_ocr/full_predictions_with_analysis.csv")
    if not bias_rows:
        print("\nGender-term bias: missing saved summary.")
        return

    by_term = {row.get("term", ""): row for row in bias_rows}
    rows = []
    for term in ["female", "women", "woman", "feminist"]:
        row = by_term.get(term, {})
        rows.append([term, row.get("count", "n/a"), format_float(row.get("fp_rate"))])

    baseline = gender_bias_baseline(prediction_rows)
    if baseline:
        rows.append(["no gendered term", baseline["count"], format_float(baseline["fp_rate"])])

    print("\nRoBERTa mixed post/OCR gender-term false-positive rates:")
    print_table(["Term", "Samples", "FP Rate"], rows)


def gender_bias_baseline(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    gendered_terms = ["woman", "women", "girl", "girls", "female", "feminist", "she", "her"]
    labels: list[int] = []
    predictions: list[int] = []
    for row in rows:
        try:
            true_label = int(row["true_label"])
            predicted_label = int(row["predicted_label"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(row.get("text") or row.get("raw_post_text") or "").lower()
        if true_label == 0 and not any(term in text for term in gendered_terms):
            labels.append(true_label)
            predictions.append(predicted_label)
    if not labels:
        return {}
    return {
        "count": len(labels),
        "fp_rate": sum(predictions) / len(predictions),
    }


def print_synthetic_diagnostic_comparison(results_dir: Path) -> None:
    baseline_overall = first_csv_row(results_dir / "synthetic_diagnostics/synthetic_summary_overall.csv")
    baseline_explicit_implicit = read_csv_rows(results_dir / "synthetic_diagnostics/synthetic_explicit_vs_implicit.csv")
    baseline_neutral = first_csv_row(results_dir / "synthetic_diagnostics/synthetic_neutral_counterexample_metrics.csv")
    baseline_emoji = first_csv_row(results_dir / "synthetic_diagnostics/emoji_pairs_summary.csv")
    augmented_overall = read_json(results_dir / "roberta_mixed_synthetic/metrics_synthetic_benchmarks_threshold_0p5.json") or {}
    augmented_diagnostics = read_json(results_dir / "roberta_mixed_synthetic/synthetic_diagnostic_metrics_threshold_0p5.json") or {}

    if not baseline_overall or not augmented_overall:
        print("\nSynthetic diagnostic benchmark comparison: missing saved summaries.")
        return

    baseline_by_source = {row.get("source_dataset", ""): row for row in baseline_explicit_implicit}
    comparison_rows = [
        [
            "Overall accuracy",
            format_float(baseline_overall.get("accuracy")),
            format_float(augmented_overall.get("accuracy")),
            format_delta(baseline_overall.get("accuracy"), augmented_overall.get("accuracy")),
        ],
        [
            "Overall macro F1",
            format_float(baseline_overall.get("macro_f1")),
            format_float(augmented_overall.get("macro_f1")),
            format_delta(baseline_overall.get("macro_f1"), augmented_overall.get("macro_f1")),
        ],
        [
            "Explicit recall",
            format_float(baseline_by_source.get("explicit", {}).get("recall")),
            format_float(augmented_diagnostics.get("explicit_recall")),
            format_delta(baseline_by_source.get("explicit", {}).get("recall"), augmented_diagnostics.get("explicit_recall")),
        ],
        [
            "Implicit recall",
            format_float(baseline_by_source.get("implicit", {}).get("recall")),
            format_float(augmented_diagnostics.get("implicit_recall")),
            format_delta(baseline_by_source.get("implicit", {}).get("recall"), augmented_diagnostics.get("implicit_recall")),
        ],
        [
            "Neutral false-positive rate",
            format_float(baseline_neutral.get("false_positive_rate")),
            format_float(augmented_diagnostics.get("neutral_false_positive_rate")),
            format_delta(baseline_neutral.get("false_positive_rate"), augmented_diagnostics.get("neutral_false_positive_rate")),
        ],
        [
            "Emoji pair flip rate",
            format_float(baseline_emoji.get("flip_rate")),
            format_float(augmented_diagnostics.get("emoji_flip_rate")),
            format_delta(baseline_emoji.get("flip_rate"), augmented_diagnostics.get("emoji_flip_rate")),
        ],
        [
            "Emoji mean delta p",
            format_float(baseline_emoji.get("mean_delta_p")),
            format_float(augmented_diagnostics.get("emoji_mean_delta_p")),
            format_delta(baseline_emoji.get("mean_delta_p"), augmented_diagnostics.get("emoji_mean_delta_p")),
        ],
    ]

    print("\nSynthetic diagnostic benchmark: RoBERTa mixed vs synthetic-augmented RoBERTa:")
    print_table(["Metric", "Mixed", "Synthetic Aug.", "Change"], comparison_rows)


def format_delta(before: Any, after: Any) -> str:
    try:
        delta = float(after) - float(before)
    except (TypeError, ValueError):
        return "n/a"
    return f"{delta:+.4f}"


def binary_metrics(labels: list[int], preds: list[int]) -> dict[str, Any]:
    n = len(labels)
    accuracy = sum(int(y == p) for y, p in zip(labels, preds)) / n if n else 0.0
    per_class: dict[str, dict[str, float | int]] = {}
    f1s = []
    weighted_sum = 0.0

    for label, name in enumerate(LABEL_NAMES):
        tp = sum(1 for y, p in zip(labels, preds) if y == label and p == label)
        fp = sum(1 for y, p in zip(labels, preds) if y != label and p == label)
        fn = sum(1 for y, p in zip(labels, preds) if y == label and p != label)
        support = sum(1 for y in labels if y == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1s.append(f1)
        weighted_sum += f1 * support

    return {
        "n_examples": n,
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s),
        "weighted_f1": weighted_sum / n if n else 0.0,
        "per_class": per_class,
    }


def import_mami_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(f"Missing base MAMI evaluation dependency: {exc}") from exc

    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("Missing open_clip. Install open_clip_torch to evaluate CLIP models.") from exc

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing transformers. Install transformers to evaluate RoBERTa+CLIP models.") from exc

    return {
        "pd": pd,
        "torch": torch,
        "nn": nn,
        "Image": Image,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "open_clip": open_clip,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
    }


def make_mami_classes(deps: dict[str, Any], device: str, max_len: int = 128):
    pd = deps["pd"]
    torch = deps["torch"]
    nn = deps["nn"]
    Image = deps["Image"]
    Dataset = deps["Dataset"]
    open_clip = deps["open_clip"]
    AutoModel = deps["AutoModel"]

    class MamiDataset(Dataset):
        def __init__(self, tsv_file: Path, image_root: Path, tokenizer=None, limit: int | None = None):
            self.df = pd.read_csv(tsv_file, sep="\t")
            if limit is not None:
                self.df = self.df.head(limit).copy()
            self.image_root = image_root
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(self.image_root / row["file_name"]).convert("RGB")
            item = {
                "image": image,
                "label": torch.tensor(int(row["label"]), dtype=torch.long),
            }
            if self.tokenizer is not None:
                encoding = self.tokenizer(
                    str(row["text"]),
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                item["input_ids"] = encoding["input_ids"].squeeze(0)
                item["attention_mask"] = encoding["attention_mask"].squeeze(0)
            return item

    def collate_fn(batch):
        out = {
            "images": [item["image"] for item in batch],
            "labels": torch.stack([item["label"] for item in batch]),
        }
        if "input_ids" in batch[0]:
            out["input_ids"] = torch.stack([item["input_ids"] for item in batch])
            out["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
        return out

    class ClipOnlyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=None
            )
            for parameter in self.clip_model.parameters():
                parameter.requires_grad = False
            self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))

        def forward(self, images):
            images = torch.stack([self.preprocess(image) for image in images]).to(device)
            image_feat = self.clip_model.encode_image(images).float()
            return self.classifier(image_feat)

    class MultimodalModel(nn.Module):
        def __init__(self, text_checkpoint: Path):
            super().__init__()
            self.text_model = AutoModel.from_pretrained(text_checkpoint)
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=None
            )
            for parameter in self.clip_model.parameters():
                parameter.requires_grad = False
            self.text_proj = nn.Linear(768, 256)
            self.image_proj = nn.Linear(512, 256)
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )

        def forward(self, input_ids, attention_mask, images):
            text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = self.text_proj(text_out.last_hidden_state[:, 0, :])
            images = torch.stack([self.preprocess(image) for image in images]).to(device)
            image_feat = self.image_proj(self.clip_model.encode_image(images).float())
            return self.classifier(torch.cat([text_feat, image_feat], dim=1))

    return MamiDataset, collate_fn, ClipOnlyModel, MultimodalModel


def mami_paths(data_dir: Path, split: str) -> tuple[Path, Path]:
    root = data_dir / "MAMI"
    tsv = root / f"{split}.tsv"
    image_dir = root / "MAMI_2022_images" / ("test_images" if split == "test" else "training_images")
    return tsv, image_dir


def evaluate_mami_model(
    spec: MamiModelSpec,
    results_dir: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    device_choice: str,
    limit: int | None,
) -> dict[str, Any]:
    deps = import_mami_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    AutoTokenizer = deps["AutoTokenizer"]

    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")

    MamiDataset, collate_fn, ClipOnlyModel, MultimodalModel = make_mami_classes(deps, device)
    tsv, image_dir = mami_paths(data_dir, split)
    if not tsv.exists():
        raise FileNotFoundError(f"Missing MAMI TSV: {tsv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing MAMI image directory: {image_dir}")

    weights_path = resolve(results_dir, spec.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights_path}")

    tokenizer = None
    if spec.model_type == "multimodal":
        if spec.text_checkpoint is None:
            raise ValueError(f"{spec.name} is multimodal but has no text checkpoint.")
        text_checkpoint = resolve(results_dir, spec.text_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(text_checkpoint)
        model = MultimodalModel(text_checkpoint)
    else:
        model = ClipOnlyModel()

    state = torch.load(weights_path, map_location=device)
    load_result = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    dataset = MamiDataset(tsv, image_dir, tokenizer=tokenizer, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    labels: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if spec.model_type == "multimodal":
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["images"],
                )
            else:
                logits = model(batch["images"])
            preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())
            if batch_index % 20 == 0:
                print(f"  {spec.name}: evaluated {min(batch_index * batch_size, len(dataset))}/{len(dataset)}")

    metrics = binary_metrics(labels, preds)
    metrics.update(
        {
            "model": spec.name,
            "split": split,
            "checkpoint": str(weights_path),
            "text_checkpoint": str(resolve(results_dir, spec.text_checkpoint)) if spec.text_checkpoint else None,
            "limit": limit,
            "state_dict_missing_keys": len(load_result.missing_keys),
            "state_dict_unexpected_keys": len(load_result.unexpected_keys),
        }
    )
    return metrics


def summarize_mami(args: argparse.Namespace) -> None:
    print_section("MAMI Multimodal Results")
    rows: list[list[Any]] = []
    missing_cache: list[MamiModelSpec] = []

    for spec in MAMI_MODELS:
        cache_path = resolve(args.results_dir, spec.cache_path)
        cached = None if args.refresh_mami_metrics else read_json(cache_path)
        if cached:
            rows.append(
                [
                    spec.name,
                    MAMI_SPLIT_LABELS.get(cached.get("split", "validation"), cached.get("split", "validation")),
                    cached.get("n_examples", "n/a"),
                    format_float(cached.get("macro_f1")),
                    format_float(cached.get("accuracy")),
                    "cached",
                ]
            )
        else:
            missing_cache.append(spec)

    if missing_cache and not args.skip_mami_eval:
        print("Computing missing MAMI metrics from saved checkpoints. No training is run.")
        for spec in missing_cache:
            print(f"\nEvaluating {spec.name}...")
            try:
                metrics = evaluate_mami_model(
                    spec,
                    args.results_dir,
                    args.data_dir,
                    args.mami_split,
                    args.batch_size,
                    args.device,
                    args.limit_mami_samples,
                )
            except (RuntimeError, FileNotFoundError) as exc:
                print(f"  Skipped: {exc}")
                continue
            cache_path = resolve(args.results_dir, spec.cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            rows.append(
                    [
                        spec.name,
                        MAMI_SPLIT_LABELS.get(metrics.get("split", args.mami_split), metrics.get("split", args.mami_split)),
                        metrics.get("n_examples", "n/a"),
                        format_float(metrics.get("macro_f1")),
                        format_float(metrics.get("accuracy")),
                    "computed",
                ]
            )

    for spec in missing_cache:
        if any(row[0] == spec.name for row in rows):
            continue
        rows.append(
            [
                spec.name,
                MAMI_SPLIT_LABELS.get(args.mami_split, args.mami_split),
                "n/a",
                "n/a",
                "n/a",
                "not evaluated",
            ]
        )

    print_table(["Model", "Evaluation Split", "N", "Macro F1", "Accuracy", "Source"], rows)
    print("\nUse --refresh-mami-metrics to recompute these from the saved .pt checkpoints.")


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    summarize_artifacts(args.results_dir)
    summarize_roberta(args.results_dir)
    summarize_error_and_diagnostics(args.results_dir)
    summarize_mami(args)


if __name__ == "__main__":
    main()
