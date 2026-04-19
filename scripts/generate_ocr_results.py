"""Run EasyOCR on post images and save a reproducible OCR CSV.

By default this reproduces the notebook behavior: scan every image file in the
dataset image directory, reuse valid cached OCR rows, run EasyOCR for anything
missing or stale, and write `.easyocr/ocr_results.csv`.
"""

from __future__ import annotations

import argparse
import ast
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
OCR_ENGINE_NAME = "easyocr"
OCR_CACHE_COLUMNS = [
    "image_id",
    "file",
    "image_path",
    "image_size_bytes",
    "image_mtime_ns",
    "ocr_engine",
    "ocr_engine_version",
    "ocr_languages",
    "ocr_text",
    "ocr_text_clean",
    "ocr_success",
    "ocr_status",
    "has_readable_text",
    "unicode_emojis_ocr_text",
    "emoticons_ocr_text",
    "laughter_tokens_ocr_text",
    "ocr_confidence_mean",
    "ocr_confidence_min",
    "ocr_detection_count",
    "processing_time_seconds",
    "error",
    "cached_at",
]
OCR_NUMERIC_COLUMNS = [
    "image_size_bytes",
    "image_mtime_ns",
    "ocr_confidence_mean",
    "ocr_confidence_min",
    "ocr_detection_count",
    "processing_time_seconds",
]
OCR_BOOL_COLUMNS = ["ocr_success", "has_readable_text"]

UNICODE_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "]"
)
OCR_EMOTICON_PATTERN = re.compile(r"(?<!\w)(?:<3|[:;=8xX][-o*']?[)(DPp/\\|])(?=\s|$|[.,!?])")
OCR_LAUGHTER_PATTERN = re.compile(r"\b(?:lol|lmao|rofl)\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "datasets" / "online-misogyny-eacl2021-main" / "data"
    easyocr_dir = project_root / ".easyocr"

    parser = argparse.ArgumentParser(
        description="Run EasyOCR on dataset post images and save an OCR-results CSV."
    )
    parser.add_argument(
        "--final-labels",
        type=Path,
        default=data_dir / "final_labels.csv",
        help="Path to final_labels.csv, used to report expected/missing image mappings.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=data_dir / "dataset_post_images",
        help="Directory containing post image files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=easyocr_dir / "ocr_results.csv",
        help="Path where OCR results should be written.",
    )
    parser.add_argument(
        "--cache-input",
        type=Path,
        default=None,
        help="Optional existing OCR CSV to use as cache. Defaults to --output.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=easyocr_dir / "model",
        help="EasyOCR model-storage directory.",
    )
    parser.add_argument(
        "--user-network-dir",
        type=Path,
        default=easyocr_dir / "user_network",
        help="EasyOCR user-network directory.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="EasyOCR language codes. Default: en.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for EasyOCR if available.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore existing OCR cache rows and recompute selected images.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Do not run EasyOCR; write cache hits and skipped rows for missing cache entries.",
    )
    parser.add_argument(
        "--selection",
        choices=["all-files", "expected-files"],
        default="all-files",
        help="OCR all image files on disk, or only files expected from final_labels.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N selected images. Useful for smoke tests.",
    )
    parser.add_argument(
        "--min-readable-chars",
        type=int,
        default=3,
        help="Minimum normalized OCR length needed for has_readable_text=True.",
    )
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def safe_int(value: object, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


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


def expected_image_file(row: pd.Series) -> str:
    image_flag = str(row.get("image", "")).strip().casefold() == "yes"
    if not image_flag:
        return ""
    top_order = parse_top_sheet_order(row.get("sheet_order", ""))
    if top_order is None:
        return ""
    return f"{row['week']}_{row['group']}_{top_order}.jpg"


def load_expected_image_names(final_labels_path: Path) -> set[str]:
    if not final_labels_path.exists():
        return set()

    labels = pd.read_csv(final_labels_path, keep_default_na=False)
    required = {"image", "sheet_order", "week", "group"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"{final_labels_path} is missing required columns: {', '.join(missing)}")

    expected = labels.apply(expected_image_file, axis=1)
    return {name for name in expected if name}


def image_files_in_dir(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )


def select_image_files(args: argparse.Namespace) -> tuple[list[Path], dict[str, int]]:
    all_image_files = image_files_in_dir(args.image_dir)
    expected_names = load_expected_image_names(args.final_labels)
    actual_names = {path.name for path in all_image_files}

    if args.selection == "expected-files":
        if not expected_names:
            raise ValueError("--selection expected-files requires a readable --final-labels CSV")
        selected = [path for path in all_image_files if path.name in expected_names]
    else:
        selected = all_image_files

    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        selected = selected[: args.limit]

    summary = {
        "image_files_found": len(all_image_files),
        "expected_image_filenames": len(expected_names),
        "missing_expected_image_files": len(expected_names - actual_names),
        "extra_image_files_not_in_labels": len(actual_names - expected_names) if expected_names else 0,
        "selected_image_files": len(selected),
    }
    return selected, summary


def normalize_ocr_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def unique_text(tokens: Iterable[object]) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        text = str(token).strip()
        if not text or text in seen:
            continue
        unique.append(text)
        seen.add(text)
    return " ".join(unique)


def find_unicode_emojis(text: object) -> list[str]:
    return UNICODE_EMOJI_PATTERN.findall(str(text))


def find_ocr_emoticons(text: object) -> list[str]:
    return [match.group(0) for match in OCR_EMOTICON_PATTERN.finditer(str(text))]


def find_ocr_laughter_tokens(text: object) -> list[str]:
    return [match.group(0) for match in OCR_LAUGHTER_PATTERN.finditer(str(text))]


def add_ocr_token_columns(results: pd.DataFrame) -> pd.DataFrame:
    if len(results) == 0:
        return results

    results = results.copy()
    text = results["ocr_text_clean"].fillna("")
    results["unicode_emojis_ocr_text"] = text.map(lambda value: unique_text(find_unicode_emojis(value)))
    results["emoticons_ocr_text"] = text.map(lambda value: unique_text(find_ocr_emoticons(value)))
    results["laughter_tokens_ocr_text"] = text.map(lambda value: unique_text(find_ocr_laughter_tokens(value)))
    return results


def project_relative_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def image_metadata(path: Path, easyocr_version: str, language_key: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "image_id": path.stem,
        "file": path.name,
        "image_path": project_relative_path(path),
        "image_size_bytes": int(stat.st_size),
        "image_mtime_ns": int(stat.st_mtime_ns),
        "ocr_engine": OCR_ENGINE_NAME,
        "ocr_engine_version": easyocr_version,
        "ocr_languages": language_key,
    }


def read_ocr_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OCR_CACHE_COLUMNS)

    cache = pd.read_csv(path, keep_default_na=False)
    for column in OCR_CACHE_COLUMNS:
        if column not in cache.columns:
            cache[column] = ""
    for column in OCR_NUMERIC_COLUMNS:
        cache[column] = pd.to_numeric(cache[column], errors="coerce")
    for column in OCR_BOOL_COLUMNS:
        cache[column] = cache[column].map(parse_bool)
    return add_ocr_token_columns(cache[OCR_CACHE_COLUMNS])


def cached_row_for_image(
    path: Path,
    cache: pd.DataFrame,
    easyocr_version: str,
    language_key: str,
    refresh: bool,
) -> dict[str, object] | None:
    if refresh or len(cache) == 0:
        return None

    metadata = image_metadata(path, easyocr_version, language_key)
    candidates = cache.loc[cache["file"].eq(metadata["file"])]
    for _, cached in candidates.iloc[::-1].iterrows():
        size_matches = safe_int(cached.get("image_size_bytes")) == metadata["image_size_bytes"]
        mtime_matches = safe_int(cached.get("image_mtime_ns")) == metadata["image_mtime_ns"]
        engine_matches = str(cached.get("ocr_engine", "")) == OCR_ENGINE_NAME
        language_matches = str(cached.get("ocr_languages", "")) == language_key
        if size_matches and mtime_matches and engine_matches and language_matches:
            row = {column: cached.get(column, "") for column in OCR_CACHE_COLUMNS}
            row["cache_hit"] = True
            return row
    return None


def empty_ocr_row(
    path: Path,
    status: str,
    easyocr_version: str,
    language_key: str,
    error: str = "",
) -> dict[str, object]:
    row = image_metadata(path, easyocr_version, language_key)
    row.update(
        {
            "ocr_text": "",
            "ocr_text_clean": "",
            "ocr_success": False,
            "ocr_status": status,
            "has_readable_text": False,
            "unicode_emojis_ocr_text": "",
            "emoticons_ocr_text": "",
            "laughter_tokens_ocr_text": "",
            "ocr_confidence_mean": np.nan,
            "ocr_confidence_min": np.nan,
            "ocr_detection_count": 0,
            "processing_time_seconds": 0.0,
            "error": error,
            "cached_at": pd.Timestamp.utcnow().isoformat(),
            "cache_hit": False,
        }
    )
    return row


def read_image_with_easyocr(path: Path, reader: object) -> tuple[str, float, float, int]:
    detections = reader.readtext(str(path), detail=1, paragraph=False)
    pieces: list[str] = []
    confidences: list[float] = []

    for detection in detections:
        if len(detection) >= 2 and str(detection[1]).strip():
            pieces.append(str(detection[1]).strip())
        if len(detection) >= 3:
            try:
                confidences.append(float(detection[2]))
            except (TypeError, ValueError):
                pass

    confidence_mean = float(np.mean(confidences)) if confidences else np.nan
    confidence_min = float(np.min(confidences)) if confidences else np.nan
    return "\n".join(pieces), confidence_mean, confidence_min, len(detections)


def run_easyocr_for_image(
    path: Path,
    reader: object,
    easyocr_version: str,
    language_key: str,
    min_readable_chars: int,
) -> dict[str, object]:
    start_time = time.perf_counter()
    try:
        text, confidence_mean, confidence_min, detection_count = read_image_with_easyocr(path, reader)
        error = ""
        ocr_success = True
    except Exception as exc:
        text = ""
        confidence_mean = np.nan
        confidence_min = np.nan
        detection_count = 0
        error = repr(exc)
        ocr_success = False

    elapsed = time.perf_counter() - start_time
    cleaned = normalize_ocr_text(text)
    has_readable_text = (
        ocr_success
        and bool(re.search(r"[A-Za-z0-9]", cleaned))
        and len(cleaned) >= min_readable_chars
    )
    row = image_metadata(path, easyocr_version, language_key)
    row.update(
        {
            "ocr_text": text,
            "ocr_text_clean": cleaned,
            "ocr_success": ocr_success,
            "ocr_status": "success" if ocr_success else "failure",
            "has_readable_text": has_readable_text,
            "unicode_emojis_ocr_text": unique_text(find_unicode_emojis(cleaned)),
            "emoticons_ocr_text": unique_text(find_ocr_emoticons(cleaned)),
            "laughter_tokens_ocr_text": unique_text(find_ocr_laughter_tokens(cleaned)),
            "ocr_confidence_mean": confidence_mean,
            "ocr_confidence_min": confidence_min,
            "ocr_detection_count": detection_count,
            "processing_time_seconds": elapsed,
            "error": error,
            "cached_at": pd.Timestamp.utcnow().isoformat(),
            "cache_hit": False,
        }
    )
    return row


def write_ocr_results(results: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    results = add_ocr_token_columns(results.drop(columns=["cache_hit"], errors="ignore"))
    for column in OCR_CACHE_COLUMNS:
        if column not in results.columns:
            results[column] = ""
    results[OCR_CACHE_COLUMNS].to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    selected_images, image_summary = select_image_files(args)

    language_key = ",".join(args.languages)
    easyocr = None
    easyocr_import_error = None
    if not args.cache_only:
        try:
            import easyocr as easyocr_module
        except ImportError as exc:
            easyocr_import_error = exc
        else:
            easyocr = easyocr_module

    easyocr_version = getattr(easyocr, "__version__", "") if easyocr is not None else ""
    cache_path = args.cache_input if args.cache_input is not None else args.output
    cache = read_ocr_cache(cache_path)
    rows: list[dict[str, object] | None] = [None] * len(selected_images)
    pending: list[tuple[int, Path]] = []
    cache_hits = 0
    computed = 0
    failures = 0

    for index, path in enumerate(selected_images):
        cached = cached_row_for_image(path, cache, easyocr_version, language_key, args.refresh)
        if cached is None:
            pending.append((index, path))
        else:
            rows[index] = cached
            cache_hits += 1

    if args.cache_only:
        for index, path in pending:
            rows[index] = empty_ocr_row(
                path,
                "skipped: cache-only and missing valid cache row",
                easyocr_version,
                language_key,
            )
    elif pending and easyocr is None:
        raise RuntimeError(
            "EasyOCR is not installed in the active environment. "
            "Install dependencies or rerun with --cache-only."
        ) from easyocr_import_error
    elif pending:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        args.user_network_dir.mkdir(parents=True, exist_ok=True)
        reader = easyocr.Reader(
            args.languages,
            gpu=args.gpu,
            model_storage_directory=str(args.model_dir),
            user_network_directory=str(args.user_network_dir),
            verbose=False,
        )
        for index, path in pending:
            row = run_easyocr_for_image(
                path,
                reader,
                easyocr_version,
                language_key,
                args.min_readable_chars,
            )
            rows[index] = row
            computed += 1
            failures += int(not row["ocr_success"])

    results = pd.DataFrame([row for row in rows if row is not None])
    for column in OCR_BOOL_COLUMNS:
        if column in results.columns:
            results[column] = results[column].map(parse_bool)
    write_ocr_results(results, args.output)

    readable = int(results["has_readable_text"].sum()) if len(results) else 0
    successes = int(results["ocr_success"].sum()) if len(results) else 0

    print(f"Image files found: {image_summary['image_files_found']:,}")
    print(f"Expected image filenames from labels: {image_summary['expected_image_filenames']:,}")
    print(f"Missing expected image files: {image_summary['missing_expected_image_files']:,}")
    print(f"Extra image files not in labels: {image_summary['extra_image_files_not_in_labels']:,}")
    print(f"Selected image files: {image_summary['selected_image_files']:,}")
    print(f"OCR cache input: {cache_path}")
    print(f"OCR cache hits: {cache_hits:,}")
    print(f"Images processed this run: {computed:,}")
    print(f"OCR successes in output: {successes:,}")
    print(f"OCR failures/skips in output: {len(results) - successes:,}")
    print(f"Images with readable OCR text: {readable:,}")
    print(f"Wrote OCR results to {args.output}")
    if failures:
        print(f"EasyOCR failures this run: {failures:,}")


if __name__ == "__main__":
    main()
