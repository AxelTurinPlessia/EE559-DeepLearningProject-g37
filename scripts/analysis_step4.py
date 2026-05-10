"""
Step 4: Analysis - Implicit vs Explicit Sexism, Bias Evaluation, Error Analysis
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
PREDICTIONS_PATH = Path("/home/lefranc/EE559-DeepLearningProject-g37/results/roberta_emoji/post_ocr_predictions.csv")
OUTPUT_DIR = Path("/home/lefranc/EE559-DeepLearningProject-g37/results/analysis_step4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load predictions ──────────────────────────────────────────────────────────
df = pd.read_csv(PREDICTIONS_PATH, keep_default_na=False)
print(f"Total samples: {len(df)}")
print(f"True positives (misogynistic): {df['true_label'].sum()}")
print(f"Overall F1 macro: {f1_score(df['true_label'], df['predicted_label'], average='macro'):.4f}")

# ── 1. Error Analysis ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. ERROR ANALYSIS")
print("="*60)

df['correct'] = df['true_label'] == df['predicted_label']
df['error_type'] = 'correct'
df.loc[(df['true_label'] == 1) & (df['predicted_label'] == 0), 'error_type'] = 'false_negative'
df.loc[(df['true_label'] == 0) & (df['predicted_label'] == 1), 'error_type'] = 'false_positive'

error_counts = df['error_type'].value_counts()
print(f"\nError breakdown:")
print(f"  Correct:        {error_counts.get('correct', 0)}")
print(f"  False Negatives (missed misogyny): {error_counts.get('false_negative', 0)}")
print(f"  False Positives (wrong alarm):     {error_counts.get('false_positive', 0)}")

# Save error examples
false_negatives = df[df['error_type'] == 'false_negative'][['post_id', 'raw_post_text', 'label']].head(20)
false_positives = df[df['error_type'] == 'false_positive'][['post_id', 'raw_post_text', 'label']].head(20)

false_negatives.to_csv(OUTPUT_DIR / "false_negatives.csv", index=False)
false_positives.to_csv(OUTPUT_DIR / "false_positives.csv", index=False)
print(f"\nSaved false negatives and false positives to {OUTPUT_DIR}")

# ── 2. Implicit vs Explicit Sexism ────────────────────────────────────────────
print("\n" + "="*60)
print("2. IMPLICIT VS EXPLICIT SEXISM ANALYSIS")
print("="*60)

# Define keywords for explicit vs implicit sexism
EXPLICIT_KEYWORDS = [
    'bitch', 'whore', 'slut', 'cunt', 'stupid woman', 'dumb woman',
    'women are', 'females are', 'hate women', 'kill women', 'rape',
    'women belong', 'go back to kitchen', 'make me a sandwich'
]

IMPLICIT_KEYWORDS = [
    'emotional', 'irrational', 'too sensitive', 'hormones', 'bossy',
    'for a woman', 'surprisingly', 'actually smart', 'not like other girls',
    'girls can\'t', 'women can\'t', 'naturally', 'biologically'
]

def classify_sexism_type(text):
    text_lower = str(text).lower()
    if any(kw in text_lower for kw in EXPLICIT_KEYWORDS):
        return 'explicit'
    elif any(kw in text_lower for kw in IMPLICIT_KEYWORDS):
        return 'implicit'
    else:
        return 'other'

# Apply only on misogynistic samples
misogynistic = df[df['true_label'] == 1].copy()
misogynistic['sexism_type'] = misogynistic['raw_post_text'].apply(classify_sexism_type)

print(f"\nSexism type distribution (in misogynistic samples):")
print(misogynistic['sexism_type'].value_counts())

# F1 per type
for stype in ['explicit', 'implicit', 'other']:
    subset = misogynistic[misogynistic['sexism_type'] == stype]
    if len(subset) > 0:
        recall = (subset['predicted_label'] == 1).mean()
        print(f"\n  {stype.upper()} ({len(subset)} samples):")
        print(f"    Recall (detected as misogynistic): {recall:.3f}")

misogynistic.to_csv(OUTPUT_DIR / "misogynistic_by_type.csv", index=False)

# ── 3. Gender Bias Analysis ───────────────────────────────────────────────────
print("\n" + "="*60)
print("3. GENDER BIAS ANALYSIS")
print("="*60)

# Test false positive rate on non-misogynistic text containing gendered terms
GENDERED_TERMS = ['woman', 'women', 'girl', 'girls', 'female', 'feminist', 'she', 'her']

non_misogynistic = df[df['true_label'] == 0].copy()

print("\nFalse positive rate by gendered term (in non-misogynistic samples):")
bias_results = []
for term in GENDERED_TERMS:
    subset = non_misogynistic[
        non_misogynistic['raw_post_text'].str.lower().str.contains(term, na=False)
    ]
    if len(subset) > 0:
        fp_rate = (subset['predicted_label'] == 1).mean()
        bias_results.append({'term': term, 'count': len(subset), 'fp_rate': fp_rate})
        print(f"  '{term}': {len(subset)} samples, FP rate = {fp_rate:.3f}")

bias_df = pd.DataFrame(bias_results)
bias_df.to_csv(OUTPUT_DIR / "gender_bias_analysis.csv", index=False)

# Overall FP rate without gendered terms (baseline)
no_gendered = non_misogynistic[
    ~non_misogynistic['raw_post_text'].str.lower().str.contains(
        '|'.join(GENDERED_TERMS), na=False
    )
]
baseline_fp = (no_gendered['predicted_label'] == 1).mean()
print(f"\n  Baseline FP rate (no gendered terms): {baseline_fp:.3f}")

# ── 4. Emoji Analysis ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. EMOJI ANALYSIS")
print("="*60)

import emoji as emoji_lib

def contains_emoji(text):
    return any(char in emoji_lib.EMOJI_DATA for char in str(text))

df['has_emoji'] = df['raw_post_text'].apply(contains_emoji)

print(f"\nSamples with emojis: {df['has_emoji'].sum()} ({df['has_emoji'].mean()*100:.1f}%)")

# F1 on samples with vs without emojis
for has_emoji in [True, False]:
    subset = df[df['has_emoji'] == has_emoji]
    if len(subset) > 0 and subset['true_label'].sum() > 0:
        f1 = f1_score(subset['true_label'], subset['predicted_label'], average='macro')
        label = "WITH emojis" if has_emoji else "WITHOUT emojis"
        print(f"\n  {label} ({len(subset)} samples):")
        print(f"    F1 macro: {f1:.4f}")
        print(f"    Misogynistic samples: {subset['true_label'].sum()}")

df.to_csv(OUTPUT_DIR / "full_predictions_with_analysis.csv", index=False)

print(f"\n✓ All analysis saved to {OUTPUT_DIR}")