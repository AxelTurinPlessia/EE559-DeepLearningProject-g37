"""Download and normalize the EDOS, Online Misogyny, and Hatemoji datasets.

The loader keeps source-specific metadata while exposing a shared text
classification schema that is convenient for training scripts:

dataset, source_repo, source_url, source_file, sample_id, split, text,
binary_label, label_text, and task.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "combined_external_datasets.csv"


@dataclass(frozen=True)
class SourceFile:
    name: str
    repo: str
    raw_url: str
    local_path: Path


SOURCE_FILES = {
    "edos": SourceFile(
        name="edos",
        repo="rewire-online/edos",
        raw_url=(
            "https://raw.githubusercontent.com/rewire-online/edos/main/"
            "data/edos_labelled_aggregated.csv"
        ),
        local_path=Path("edos/data/edos_labelled_aggregated.csv"),
    ),
    "online_misogyny": SourceFile(
        name="online_misogyny",
        repo="ellamguest/online-misogyny-eacl2021",
        raw_url=(
            "https://raw.githubusercontent.com/ellamguest/online-misogyny-eacl2021/"
            "main/data/final_labels.csv"
        ),
        local_path=Path("online-misogyny-eacl2021-main/data/final_labels.csv"),
    ),
    "hatemoji_build_train": SourceFile(
        name="hatemoji_build_train",
        repo="HannahKirk/Hatemoji",
        raw_url=(
            "https://raw.githubusercontent.com/HannahKirk/Hatemoji/main/"
            "HatemojiBuild/train.csv"
        ),
        local_path=Path("Hatemoji/HatemojiBuild/train.csv"),
    ),
    "hatemoji_build_validation": SourceFile(
        name="hatemoji_build_validation",
        repo="HannahKirk/Hatemoji",
        raw_url=(
            "https://raw.githubusercontent.com/HannahKirk/Hatemoji/main/"
            "HatemojiBuild/validation.csv"
        ),
        local_path=Path("Hatemoji/HatemojiBuild/validation.csv"),
    ),
    "hatemoji_build_test": SourceFile(
        name="hatemoji_build_test",
        repo="HannahKirk/Hatemoji",
        raw_url=(
            "https://raw.githubusercontent.com/HannahKirk/Hatemoji/main/"
            "HatemojiBuild/test.csv"
        ),
        local_path=Path("Hatemoji/HatemojiBuild/test.csv"),
    ),
    "hatemoji_check_test": SourceFile(
        name="hatemoji_check_test",
        repo="HannahKirk/Hatemoji",
        raw_url=(
            "https://raw.githubusercontent.com/HannahKirk/Hatemoji/main/"
            "HatemojiCheck/test.csv"
        ),
        local_path=Path("Hatemoji/HatemojiCheck/test.csv"),
    ),
}

DATASET_CHOICES = ("edos", "online_misogyny", "hatemoji_build", "hatemoji_check")
COMMON_COLUMNS = [
    "dataset",
    "source_repo",
    "source_url",
    "source_file",
    "sample_id",
    "split",
    "text",
    "binary_label",
    "label_text",
    "task",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and normalize EDOS, Online Misogyny, HatemojiBuild, "
            "and HatemojiCheck into a shared CSV schema."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory where source CSVs are cached.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the combined normalized CSV.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_CHOICES,
        default=list(DATASET_CHOICES),
        help="Datasets to load. Default: all.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing CSVs; require them under --data-root.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload source CSVs even if local copies already exist.",
    )
    parser.set_defaults(drop_empty_text=True, drop_unlabeled=True)
    parser.add_argument(
        "--keep-empty-text",
        action="store_false",
        dest="drop_empty_text",
        help="Keep rows where the normalized text field is empty. Default: dropped.",
    )
    parser.add_argument(
        "--keep-unlabeled",
        action="store_false",
        dest="drop_unlabeled",
        help="Keep rows that cannot be mapped to a binary label. Default: dropped.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print per-dataset/split/label counts after loading.",
    )
    return parser.parse_args()


def source_path(data_root: Path, source: SourceFile) -> Path:
    return data_root / source.local_path


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "EE559-dataset-loader"})
    try:
        with urlopen(request, timeout=60) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def ensure_sources(
    data_root: Path,
    source_names: list[str],
    download_missing: bool = True,
    refresh: bool = False,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in source_names:
        source = SOURCE_FILES[name]
        path = source_path(data_root, source)
        if refresh or not path.exists():
            if not download_missing and not refresh:
                raise FileNotFoundError(
                    f"Missing {path}. Re-run without --no-download to fetch it."
                )
            print(f"Downloading {source.repo}:{source.local_path} -> {path}")
            download_file(source.raw_url, path)
        paths[name] = path
    return paths


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def clean_text(value: object) -> str:
    return " ".join(str(value).split()) if pd.notna(value) else ""


def add_common_metadata(
    df: pd.DataFrame,
    dataset: str,
    source: SourceFile,
    task: str,
) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = dataset
    df["source_repo"] = source.repo
    df["source_url"] = source.raw_url
    df["source_file"] = str(source.local_path)
    df["task"] = task
    return df


def normalize_binary_label(
    label: pd.Series,
    positive_value: object,
    positive_name: str,
    negative_name: str,
) -> tuple[pd.Series, pd.Series]:
    binary = (label == positive_value).astype("Int64")
    label_text = binary.map({1: positive_name, 0: negative_name})
    return binary, label_text


def load_edos(data_root: Path, download_missing: bool = True, refresh: bool = False) -> pd.DataFrame:
    paths = ensure_sources(data_root, ["edos"], download_missing, refresh)
    source = SOURCE_FILES["edos"]
    path = paths["edos"]
    df = pd.read_csv(path, keep_default_na=False)
    require_columns(df, path, {"rewire_id", "text", "label_sexist", "split"})

    out = pd.DataFrame(
        {
            "sample_id": df["rewire_id"].astype(str),
            "split": df["split"].astype(str),
            "text": df["text"].map(clean_text),
            "label_category": df.get("label_category", ""),
            "label_vector": df.get("label_vector", ""),
        }
    )
    out["binary_label"], out["label_text"] = normalize_binary_label(
        df["label_sexist"], "sexist", "sexist", "not sexist"
    )
    out = add_common_metadata(out, "edos", source, "sexism")
    return out


def load_hatemoji_build(
    data_root: Path,
    download_missing: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    source_names = [
        "hatemoji_build_train",
        "hatemoji_build_validation",
        "hatemoji_build_test",
    ]
    paths = ensure_sources(data_root, source_names, download_missing, refresh)
    frames = []

    for name in source_names:
        source = SOURCE_FILES[name]
        path = paths[name]
        df = pd.read_csv(path, keep_default_na=False)
        require_columns(df, path, {"entry_id", "text", "split", "label_gold"})
        split_from_file = name.removeprefix("hatemoji_build_")

        out = pd.DataFrame(
            {
                "sample_id": df["entry_id"].astype(str),
                "split": df["split"].astype(str).where(df["split"].astype(str).ne(""), split_from_file),
                "text": df["text"].map(clean_text),
                "hate_type": df.get("type", ""),
                "target": df.get("target", ""),
                "round_base": df.get("round.base", ""),
                "round_set": df.get("round.set", ""),
                "set": df.get("set", ""),
                "matched_text": df.get("matched_text", ""),
                "matched_id": df.get("matched_id", ""),
            }
        )
        out["binary_label"], out["label_text"] = normalize_binary_label(
            df["label_gold"].astype(int), 1, "hateful", "non-hateful"
        )
        frames.append(add_common_metadata(out, "hatemoji_build", source, "hate"))

    return pd.concat(frames, ignore_index=True)


def load_hatemoji_check(
    data_root: Path,
    download_missing: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    paths = ensure_sources(data_root, ["hatemoji_check_test"], download_missing, refresh)
    source = SOURCE_FILES["hatemoji_check_test"]
    path = paths["hatemoji_check_test"]
    df = pd.read_csv(path, keep_default_na=False)
    require_columns(df, path, {"case_id", "text", "label_gold"})

    out = pd.DataFrame(
        {
            "sample_id": df["case_id"].astype(str),
            "split": "test",
            "text": df["text"].map(clean_text),
            "target": df.get("target", ""),
            "functionality": df.get("functionality", ""),
            "set": df.get("set", ""),
            "unrealistic_flags": df.get("unrealistic_flags", ""),
            "included_in_test_suite": df.get("included_in_test_suite", ""),
        }
    )
    out["binary_label"], out["label_text"] = normalize_binary_label(
        df["label_gold"].astype(int), 1, "hateful", "non-hateful"
    )
    return add_common_metadata(out, "hatemoji_check", source, "hate")


def parse_top_sheet_order(value: object) -> int | None:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, tuple) or not parsed:
        return None
    try:
        return int(parsed[0])
    except (TypeError, ValueError):
        return None


def raw_text_from_online_misogyny_row(row: pd.Series) -> str:
    body = str(row.get("body", "")).strip()
    image_value = str(row.get("image", "")).strip()
    if image_value and image_value.casefold() != "yes":
        return clean_text(f"{body}, {image_value}" if body else image_value)
    return clean_text(body)


def expected_image_file(row: pd.Series) -> str:
    image_value = str(row.get("image", "")).strip().casefold()
    if image_value != "yes":
        return ""
    top_order = parse_top_sheet_order(row.get("sheet_order", ""))
    if top_order is None:
        return ""
    return f"{row['week']}_{row['group']}_{top_order}.jpg"


def unique_nonempty(values: pd.Series) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def merge_unique_text(values: pd.Series) -> str:
    return "\n".join(unique_nonempty(values))


def collapse_label(values: pd.Series) -> str:
    labels = unique_nonempty(values)
    if len(labels) == 1:
        return labels[0]
    return "Conflict: " + " + ".join(sorted(labels))


def collapse_split(values: pd.Series) -> str:
    splits = unique_nonempty(values)
    if len(splits) == 1:
        return splits[0]
    return "+".join(sorted(splits))


def load_online_misogyny(
    data_root: Path,
    download_missing: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    paths = ensure_sources(data_root, ["online_misogyny"], download_missing, refresh)
    source = SOURCE_FILES["online_misogyny"]
    path = paths["online_misogyny"]
    df = pd.read_csv(path, keep_default_na=False)
    required = {"entry_id", "body", "image", "week", "group", "sheet_order", "level_1", "split"}
    require_columns(df, path, required)

    df = df.copy()
    df["normalized_text"] = df.apply(raw_text_from_online_misogyny_row, axis=1)
    df["expected_image_file"] = df.apply(expected_image_file, axis=1)
    image_dir = path.parent / "dataset_post_images"

    grouped = (
        df.groupby("entry_id", sort=False)
        .agg(
            text=("normalized_text", merge_unique_text),
            label_text=("level_1", collapse_label),
            split=("split", collapse_split),
            image_files=("expected_image_file", unique_nonempty),
            subreddit=("subreddit", collapse_split),
        )
        .reset_index()
        .rename(columns={"entry_id": "sample_id"})
    )
    grouped["image_files"] = grouped["image_files"].map(lambda files: "|".join(files))
    grouped["image_exists"] = grouped["image_files"].map(
        lambda value: any((image_dir / name).exists() for name in value.split("|") if name)
    )
    grouped["binary_label"] = grouped["label_text"].map(
        {"Nonmisogynistic": 0, "Misogynistic": 1}
    ).astype("Int64")
    return add_common_metadata(grouped, "online_misogyny", source, "misogyny")


def load_selected_datasets(
    dataset_names: list[str],
    data_root: Path = DEFAULT_DATA_ROOT,
    download_missing: bool = True,
    refresh: bool = False,
    drop_empty_text: bool = True,
    drop_unlabeled: bool = True,
) -> dict[str, pd.DataFrame]:
    loaders = {
        "edos": load_edos,
        "online_misogyny": load_online_misogyny,
        "hatemoji_build": load_hatemoji_build,
        "hatemoji_check": load_hatemoji_check,
    }
    loaded = {
        name: loaders[name](data_root, download_missing=download_missing, refresh=refresh)
        for name in dataset_names
    }
    if drop_empty_text:
        loaded = {
            name: df[df["text"].astype(str).str.strip().ne("")].copy()
            for name, df in loaded.items()
        }
    if drop_unlabeled:
        loaded = {
            name: df[df["binary_label"].notna()].copy()
            for name, df in loaded.items()
        }
    return loaded


def combine_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(datasets.values(), ignore_index=True, sort=False)
    front = [column for column in COMMON_COLUMNS if column in combined.columns]
    rest = [column for column in combined.columns if column not in front]
    return combined[front + rest]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "split", "label_text"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["dataset", "split", "label_text"])
    )


def main() -> None:
    args = parse_args()
    selected = list(dict.fromkeys(args.datasets))
    loaded = load_selected_datasets(
        selected,
        data_root=args.data_root,
        download_missing=not args.no_download,
        refresh=args.refresh,
        drop_empty_text=args.drop_empty_text,
        drop_unlabeled=args.drop_unlabeled,
    )
    combined = combine_datasets(loaded)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined):,} rows to {args.output}")

    if args.print_summary:
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(summary_table(combined).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
