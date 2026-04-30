"""Create mixed RoBERTa train/validation/test CSVs.

The mixed dataset combines the current RoBERTa text sources
(HatemojiBuild train+validation and EDOS train+dev) with a stratified split of
the post/OCR dataset. The final test split is kept post/OCR-only so it remains
a clean estimate of performance on the target dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Iterable

try:
    import emoji
except ModuleNotFoundError:  # pragma: no cover - depends on the active environment.
    emoji = None


LABEL_TEXT = {
    0: "non-misogynistic",
    1: "misogynistic",
}

OUTPUT_COLUMNS = [
    "sample_id",
    "split",
    "source",
    "source_split",
    "text",
    "binary_label",
    "label_text",
    "post_id",
    "raw_post_text",
    "ocr_text",
    "ocr_readable",
]


Row = dict[str, str | int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mixed RoBERTa/post-OCR train, validation, and test splits."
    )
    parser.add_argument(
        "--hatemoji_dir",
        type=Path,
        default=Path("datasets/Hatemoji/HatemojiBuild"),
        help="Directory containing HatemojiBuild train.csv and validation.csv.",
    )
    parser.add_argument(
        "--edos_path",
        type=Path,
        default=Path("datasets/edos/data/edos_labelled_aggregated.csv"),
        help="Path to EDOS aggregated labels CSV.",
    )
    parser.add_argument(
        "--post_ocr_path",
        type=Path,
        default=Path("datasets/post_ocr_dataset.csv"),
        help="Path to the merged post/OCR dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets/roberta_mixed_post_ocr"),
        help="Directory where split CSVs and summary metadata are written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--original_val_size",
        type=float,
        default=0.15,
        help="Validation fraction for Hatemoji+EDOS sources.",
    )
    parser.add_argument(
        "--post_ocr_train_size",
        type=float,
        default=0.70,
        help="Training fraction for post/OCR examples.",
    )
    parser.add_argument(
        "--post_ocr_val_size",
        type=float,
        default=0.15,
        help="Validation fraction for post/OCR examples.",
    )
    parser.add_argument(
        "--post_ocr_test_size",
        type=float,
        default=0.15,
        help="Held-out test fraction for post/OCR examples.",
    )
    return parser.parse_args()


def process_emojis(text: object) -> str:
    if not isinstance(text, str):
        return ""
    if emoji is None:
        return text
    return emoji.demojize(text, delimiters=(" :", ": "))


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = process_emojis(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_post_text(row: Row) -> str:
    parts = [str(row.get("raw_post_text", "")).strip()]
    ocr_text = str(row.get("ocr_text", "")).strip()
    if ocr_text and ocr_text not in parts:
        parts.append(ocr_text)
    return "\n".join(part for part in parts if part)


def read_csv(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def common_row(row: Row) -> Row:
    clean = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
    clean["binary_label"] = int(clean["binary_label"])
    clean["label_text"] = LABEL_TEXT[int(clean["binary_label"])]
    return clean


def load_hatemoji(hatemoji_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for filename, source_split in [
        ("train.csv", "train"),
        ("validation.csv", "validation"),
    ]:
        for row in read_csv(hatemoji_dir / filename):
            rows.append(
                common_row(
                    {
                        "sample_id": f"hatemoji_{row['entry_id']}",
                        "source": "hatemoji",
                        "source_split": source_split,
                        "text": clean_text(row["text"]),
                        "binary_label": int(row["label_gold"]),
                    }
                )
            )
    return rows


def load_edos(edos_path: Path) -> list[Row]:
    rows: list[Row] = []
    for row in read_csv(edos_path):
        if row["split"] not in {"train", "dev"}:
            continue
        rows.append(
            common_row(
                {
                    "sample_id": f"edos_{row['rewire_id']}",
                    "source": "edos",
                    "source_split": row["split"],
                    "text": clean_text(row["text"]),
                    "binary_label": int(row["label_sexist"] == "sexist"),
                }
            )
        )
    return rows


def load_original_sources(hatemoji_dir: Path, edos_path: Path) -> list[Row]:
    rows = load_hatemoji(hatemoji_dir) + load_edos(edos_path)
    return [row for row in rows if str(row["text"]).strip()]


def load_post_ocr(post_ocr_path: Path) -> list[Row]:
    label_map = {"Nonmisogynistic": 0, "Misogynistic": 1}
    rows: list[Row] = []
    for row in read_csv(post_ocr_path):
        if row["label"] not in label_map:
            continue
        text = clean_text(combine_post_text(row))
        if not text:
            continue
        rows.append(
            common_row(
                {
                    "sample_id": f"post_ocr_{row['post_id']}",
                    "source": "post_ocr",
                    "source_split": "full",
                    "text": text,
                    "binary_label": label_map[row["label"]],
                    "post_id": row["post_id"],
                    "raw_post_text": row["raw_post_text"],
                    "ocr_text": row["ocr_text"],
                    "ocr_readable": row["ocr_readable"],
                }
            )
        )
    return rows


def validate_fractions(args: argparse.Namespace) -> None:
    post_total = args.post_ocr_train_size + args.post_ocr_val_size + args.post_ocr_test_size
    if not math.isclose(post_total, 1.0):
        raise ValueError(
            "Post/OCR fractions must sum to 1.0: "
            f"got {args.post_ocr_train_size} + {args.post_ocr_val_size} "
            f"+ {args.post_ocr_test_size} = {post_total}"
        )
    for name in [
        "original_val_size",
        "post_ocr_train_size",
        "post_ocr_val_size",
        "post_ocr_test_size",
    ]:
        value = getattr(args, name)
        if value <= 0 or value >= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")


def split_quotas(group_sizes: dict[int, int], holdout_size: int, holdout_fraction: float) -> dict[int, int]:
    raw_quotas = {
        label: group_size * holdout_fraction
        for label, group_size in group_sizes.items()
    }
    quotas = {label: int(math.floor(raw)) for label, raw in raw_quotas.items()}
    remaining = holdout_size - sum(quotas.values())
    labels_by_remainder = sorted(
        raw_quotas,
        key=lambda label: (raw_quotas[label] - quotas[label], group_sizes[label]),
        reverse=True,
    )
    for label in labels_by_remainder[:remaining]:
        quotas[label] += 1
    return quotas


def stratified_holdout_split(rows: list[Row], holdout_fraction: float, seed: int) -> tuple[list[Row], list[Row]]:
    if not rows:
        return [], []

    groups: dict[int, list[Row]] = {}
    for row in rows:
        groups.setdefault(int(row["binary_label"]), []).append(row)

    holdout_size = int(math.ceil(len(rows) * holdout_fraction))
    quotas = split_quotas(
        {label: len(label_rows) for label, label_rows in groups.items()},
        holdout_size,
        holdout_fraction,
    )

    rng = random.Random(seed)
    train_rows: list[Row] = []
    holdout_rows: list[Row] = []
    for label in sorted(groups):
        label_rows = groups[label][:]
        rng.shuffle(label_rows)
        split_at = quotas[label]
        holdout_rows.extend(label_rows[:split_at])
        train_rows.extend(label_rows[split_at:])

    return train_rows, holdout_rows


def split_original(rows: list[Row], val_size: float, seed: int) -> tuple[list[Row], list[Row]]:
    return stratified_holdout_split(rows, val_size, seed)


def split_post_ocr(
    rows: list[Row],
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[list[Row], list[Row], list[Row]]:
    train_val, test_rows = stratified_holdout_split(rows, test_size, seed)
    relative_val_size = val_size / (1.0 - test_size)
    train_rows, val_rows = stratified_holdout_split(train_val, relative_val_size, seed)
    return train_rows, val_rows, test_rows


def assign_split(rows: list[Row], split: str) -> list[Row]:
    assigned: list[Row] = []
    for row in rows:
        copied = row.copy()
        copied["split"] = split
        assigned.append(copied)
    return assigned


def shuffle_rows(rows: list[Row], seed: int) -> list[Row]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled


def class_counts(rows: list[Row]) -> dict[str, int]:
    counts = {label_text: 0 for label_text in LABEL_TEXT.values()}
    for row in rows:
        counts[LABEL_TEXT[int(row["binary_label"])]] += 1
    return {label_text: count for label_text, count in counts.items() if count}


def source_counts(rows: list[Row]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row["source"])
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def split_summary(rows: list[Row]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "class_counts": class_counts(rows),
        "source_counts": source_counts(rows),
    }


def ensure_no_post_ocr_leakage(train: list[Row], val: list[Row], test: list[Row]) -> None:
    post_ids_by_split = {
        "train": {str(row["post_id"]) for row in train if row["source"] == "post_ocr"},
        "val": {str(row["post_id"]) for row in val if row["source"] == "post_ocr"},
        "test": {str(row["post_id"]) for row in test if row["source"] == "post_ocr"},
    }
    overlaps = {
        "train_val": post_ids_by_split["train"] & post_ids_by_split["val"],
        "train_test": post_ids_by_split["train"] & post_ids_by_split["test"],
        "val_test": post_ids_by_split["val"] & post_ids_by_split["test"],
    }
    leaked = {name: sorted(values)[:10] for name, values in overlaps.items() if values}
    if leaked:
        raise ValueError(f"Post/OCR post IDs overlap across final splits: {leaked}")


def main() -> None:
    args = parse_args()
    validate_fractions(args)
    if emoji is None:
        print("Warning: emoji package is not installed; keeping emoji characters unchanged.")

    original = load_original_sources(args.hatemoji_dir, args.edos_path)
    post_ocr = load_post_ocr(args.post_ocr_path)

    original_train, original_val = split_original(original, args.original_val_size, args.seed)
    post_train, post_val, post_test = split_post_ocr(
        post_ocr,
        val_size=args.post_ocr_val_size,
        test_size=args.post_ocr_test_size,
        seed=args.seed,
    )

    train = shuffle_rows(
        assign_split(original_train, "train") + assign_split(post_train, "train"),
        args.seed,
    )
    val = shuffle_rows(
        assign_split(original_val, "val") + assign_split(post_val, "val"),
        args.seed,
    )
    test = shuffle_rows(assign_split(post_test, "test"), args.seed)

    ensure_no_post_ocr_leakage(train, val, test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "train.csv", train)
    write_csv(args.output_dir / "val.csv", val)
    write_csv(args.output_dir / "test.csv", test)
    write_csv(args.output_dir / "all.csv", train + val + test)

    summary = {
        "seed": args.seed,
        "emoji_processing": "demojize" if emoji is not None else "unchanged_missing_emoji_package",
        "source_paths": {
            "hatemoji_dir": str(args.hatemoji_dir),
            "edos_path": str(args.edos_path),
            "post_ocr_path": str(args.post_ocr_path),
        },
        "fractions": {
            "original_train": 1.0 - args.original_val_size,
            "original_val": args.original_val_size,
            "post_ocr_train": args.post_ocr_train_size,
            "post_ocr_val": args.post_ocr_val_size,
            "post_ocr_test": args.post_ocr_test_size,
        },
        "splits": {
            "train": split_summary(train),
            "val": split_summary(val),
            "test": split_summary(test),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote mixed dataset to {args.output_dir}")
    for split_name, rows in [("train", train), ("val", val), ("test", test)]:
        print(
            f"{split_name}: {len(rows):,} rows | "
            f"classes={class_counts(rows)} | sources={source_counts(rows)}"
        )


if __name__ == "__main__":
    main()
