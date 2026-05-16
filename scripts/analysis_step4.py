"""
Step 4: Analysis - Implicit vs Explicit Sexism, Bias Evaluation, Error Analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

DEFAULT_RUN_DIR = Path("results/roberta_mixed_post_ocr/lr2e-5_seed42")
DEFAULT_OUTPUT_DIR = Path("results/analysis_step4_roberta_mixed_post_ocr")

EXPLICIT_KEYWORDS = [
    "bitch",
    "whore",
    "slut",
    "cunt",
    "stupid woman",
    "dumb woman",
    "women are",
    "females are",
    "hate women",
    "kill women",
    "rape",
    "women belong",
    "go back to kitchen",
    "make me a sandwich",
]

IMPLICIT_KEYWORDS = [
    "emotional",
    "irrational",
    "too sensitive",
    "hormones",
    "bossy",
    "for a woman",
    "surprisingly",
    "actually smart",
    "not like other girls",
    "girls can't",
    "women can't",
    "naturally",
    "biologically",
]

GENDERED_TERMS = ["woman", "women", "girl", "girls", "female", "feminist", "she", "her"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step 4 error, sexism-type, gender-bias, and emoji analyses."
    )
    parser.add_argument(
        "--predictions_path",
        type=Path,
        default=DEFAULT_RUN_DIR / "test_predictions.csv",
        help="CSV containing true_label, predicted_label, and text columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where analysis CSVs are written.",
    )
    parser.add_argument(
        "--text_column",
        default="text",
        help="Text column to analyze. Falls back to raw_post_text when absent.",
    )
    return parser.parse_args()


def pick_text_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    if "raw_post_text" in df.columns:
        print(f"Text column '{requested}' not found; using 'raw_post_text'.")
        return "raw_post_text"
    raise ValueError(f"Missing requested text column '{requested}' and fallback 'raw_post_text'.")


def classify_sexism_type(text: object) -> str:
    text_lower = str(text).lower()
    if any(kw in text_lower for kw in EXPLICIT_KEYWORDS):
        return "explicit"
    if any(kw in text_lower for kw in IMPLICIT_KEYWORDS):
        return "implicit"
    return "other"


def contains_emoji(text: object) -> bool:
    import emoji as emoji_lib

    return any(char in emoji_lib.EMOJI_DATA for char in str(text))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_path, keep_default_na=False)
    text_column = pick_text_column(df, args.text_column)
    print(f"Predictions: {args.predictions_path}")
    print(f"Text column: {text_column}")
    print(f"Total samples: {len(df)}")
    print(f"True positives (misogynistic): {df['true_label'].sum()}")
    print(f"Overall F1 macro: {f1_score(df['true_label'], df['predicted_label'], average='macro'):.4f}")

    # ── 1. Error Analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. ERROR ANALYSIS")
    print("=" * 60)

    df["correct"] = df["true_label"] == df["predicted_label"]
    df["error_type"] = "correct"
    df.loc[(df["true_label"] == 1) & (df["predicted_label"] == 0), "error_type"] = "false_negative"
    df.loc[(df["true_label"] == 0) & (df["predicted_label"] == 1), "error_type"] = "false_positive"

    error_counts = df["error_type"].value_counts()
    print("\nError breakdown:")
    print(f"  Correct:        {error_counts.get('correct', 0)}")
    print(f"  False Negatives (missed misogyny): {error_counts.get('false_negative', 0)}")
    print(f"  False Positives (wrong alarm):     {error_counts.get('false_positive', 0)}")

    error_columns = [
        column
        for column in ["post_id", "sample_id", text_column, "raw_post_text", "ocr_text", "label_text"]
        if column in df.columns
    ]
    false_negatives = df[df["error_type"] == "false_negative"][error_columns].head(20)
    false_positives = df[df["error_type"] == "false_positive"][error_columns].head(20)

    false_negatives.to_csv(args.output_dir / "false_negatives.csv", index=False)
    false_positives.to_csv(args.output_dir / "false_positives.csv", index=False)
    print(f"\nSaved false negatives and false positives to {args.output_dir}")

    # ── 2. Implicit vs Explicit Sexism ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. IMPLICIT VS EXPLICIT SEXISM ANALYSIS")
    print("=" * 60)

    misogynistic = df[df["true_label"] == 1].copy()
    misogynistic["sexism_type"] = misogynistic[text_column].apply(classify_sexism_type)

    print("\nSexism type distribution (in misogynistic samples):")
    print(misogynistic["sexism_type"].value_counts())

    sexism_type_results = []
    for stype in ["explicit", "implicit", "other"]:
        subset = misogynistic[misogynistic["sexism_type"] == stype]
        if len(subset) > 0:
            recall = (subset["predicted_label"] == 1).mean()
            sexism_type_results.append({"sexism_type": stype, "count": len(subset), "recall": recall})
            print(f"\n  {stype.upper()} ({len(subset)} samples):")
            print(f"    Recall (detected as misogynistic): {recall:.3f}")

    misogynistic.to_csv(args.output_dir / "misogynistic_by_type.csv", index=False)
    pd.DataFrame(sexism_type_results).to_csv(
        args.output_dir / "sexism_type_summary.csv",
        index=False,
    )

    # ── 3. Gender Bias Analysis ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. GENDER BIAS ANALYSIS")
    print("=" * 60)

    non_misogynistic = df[df["true_label"] == 0].copy()

    print("\nFalse positive rate by gendered term (in non-misogynistic samples):")
    bias_results = []
    text_lower = non_misogynistic[text_column].astype(str).str.lower()
    for term in GENDERED_TERMS:
        subset = non_misogynistic[text_lower.str.contains(term, na=False, regex=False)]
        if len(subset) > 0:
            fp_rate = (subset["predicted_label"] == 1).mean()
            bias_results.append({"term": term, "count": len(subset), "fp_rate": fp_rate})
            print(f"  '{term}': {len(subset)} samples, FP rate = {fp_rate:.3f}")

    bias_df = pd.DataFrame(bias_results)
    bias_df.to_csv(args.output_dir / "gender_bias_analysis.csv", index=False)

    gender_pattern = "|".join(re.escape(term) for term in GENDERED_TERMS)
    no_gendered = non_misogynistic[~text_lower.str.contains(gender_pattern, na=False, regex=True)]
    baseline_fp = (no_gendered["predicted_label"] == 1).mean()
    print(f"\n  Baseline FP rate (no gendered terms): {baseline_fp:.3f}")

    # ── 4. Emoji Analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. EMOJI ANALYSIS")
    print("=" * 60)

    emoji_column = "raw_post_text" if "raw_post_text" in df.columns else text_column
    df["has_emoji"] = df[emoji_column].apply(contains_emoji)

    print(f"\nSamples with emojis: {df['has_emoji'].sum()} ({df['has_emoji'].mean() * 100:.1f}%)")

    emoji_results = []
    for has_emoji in [True, False]:
        subset = df[df["has_emoji"] == has_emoji]
        if len(subset) > 0 and subset["true_label"].sum() > 0:
            f1 = f1_score(subset["true_label"], subset["predicted_label"], average="macro")
            label = "WITH emojis" if has_emoji else "WITHOUT emojis"
            emoji_results.append(
                {
                    "has_emoji": has_emoji,
                    "count": len(subset),
                    "misogynistic_count": int(subset["true_label"].sum()),
                    "macro_f1": f1,
                }
            )
            print(f"\n  {label} ({len(subset)} samples):")
            print(f"    F1 macro: {f1:.4f}")
            print(f"    Misogynistic samples: {subset['true_label'].sum()}")

    pd.DataFrame(emoji_results).to_csv(args.output_dir / "emoji_summary.csv", index=False)
    df.to_csv(args.output_dir / "full_predictions_with_analysis.csv", index=False)

    print(f"\nAll analysis saved to {args.output_dir}")


if __name__ == "__main__":
    main()
