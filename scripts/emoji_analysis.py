"""
Step 4: Controlled examples - testing emoji influence on model predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import emoji
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_PATH = Path("results/roberta_mixed_post_ocr/lr2e-5_seed42/best_model")
DEFAULT_OUTPUT_PATH = Path(
    "results/analysis_step4_roberta_mixed_post_ocr/controlled_examples_results.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled emoji-pair predictions for a sequence classifier."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Model checkpoint directory to evaluate.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path for controlled-pair results.",
    )
    return parser.parse_args()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text, tokenizer, model, device):
    cleaned = clean_text(text)
    enc = tokenizer(cleaned, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = model(input_ids=enc['input_ids'].to(device),
                    attention_mask=enc['attention_mask'].to(device))
    prob = torch.softmax(out.logits, dim=1)[0]
    pred = torch.argmax(out.logits, dim=1).item()
    return pred, prob[1].item()  # pred, probability of misogynistic

def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    # ── Controlled pairs ──────────────────────────────────────────────────────
    pairs = [
    # (text_without_emoji, text_with_emoji, category)
    ("Women are so smart", "Women are so smart 🙄", "sarcasm"),
    ("She did a great job", "She did a great job... for a woman 😂", "sarcasm"),
    ("Women should be respected", "Women should be respected 😏", "sarcasm"),
    ("Girls can do anything", "Girls can do anything 🤡", "sarcasm"),
    ("I love working with women", "I love working with women 😒", "sarcasm"),
    ("She's very emotional", "She's very emotional 🙄", "stereotype"),
    ("Women are naturally nurturing", "Women are naturally nurturing 💅", "stereotype"),
    ("She belongs in the kitchen", "She belongs in the kitchen 👩‍🍳", "explicit"),
    ("Women are too sensitive", "Women are too sensitive 😢", "stereotype"),
    ("She's bossy", "She's bossy 😤", "stereotype"),
    ("Girls aren't good at math", "Girls aren't good at math 📉", "stereotype"),
    ("Women can't drive", "Women can't drive 🚗💥", "stereotype"),
    ("She got promoted? Must be diversity hire", "She got promoted? Must be diversity hire 🤷", "implicit"),
    ("Women always overreact", "Women always overreact 😱", "stereotype"),
    ("She's pretty smart for a girl", "She's pretty smart for a girl ✨", "backhanded"),
    ("Not bad for a woman", "Not bad for a woman 👏", "backhanded"),
    ("Women are so dramatic", "Women are so dramatic 🎭", "stereotype"),
    ("She's surprisingly competent", "She's surprisingly competent 😮", "backhanded"),
    ("Girls should smile more", "Girls should smile more 😊", "condescending"),
    ("Women need to calm down", "Women need to calm down 🧘", "condescending"),
    ]

    # ── Run predictions ───────────────────────────────────────────────────────
    results = []
    flipped = 0

    print(f"{'Category':<15} {'Prediction change':<20} {'Text'}")
    print("-" * 80)

    for text_plain, text_emoji, category in pairs:
        pred_plain, prob_plain = predict(text_plain, tokenizer, model, device)
        pred_emoji, prob_emoji = predict(text_emoji, tokenizer, model, device)

        flipped_this = pred_plain != pred_emoji
        if flipped_this:
            flipped += 1

        label_plain = "Misogynistic" if pred_plain == 1 else "Non-misogynistic"
        label_emoji = "Misogynistic" if pred_emoji == 1 else "Non-misogynistic"
        change = "FLIP" if flipped_this else "same"

        print(f"{category:<15} {change:<20} '{text_plain[:40]}'")
        print(f"               Plain: {label_plain} (p={prob_plain:.3f})")
        print(f"               Emoji: {label_emoji} (p={prob_emoji:.3f})")
        print()

        results.append({
            'text_plain': text_plain,
            'text_with_emoji': text_emoji,
            'category': category,
            'pred_plain': label_plain,
            'pred_emoji': label_emoji,
            'prob_plain': round(prob_plain, 3),
            'prob_emoji': round(prob_emoji, 3),
            'prediction_flipped': flipped_this,
        })

    print(f"\n=== SUMMARY ===")
    print(f"Total pairs: {len(pairs)}")
    print(f"Predictions flipped by emoji: {flipped} / {len(pairs)} ({flipped/len(pairs)*100:.1f}%)")

    df = pd.DataFrame(results)
    print(f"\nFlips by category:")
    print(df.groupby('category')['prediction_flipped'].sum())

    df.to_csv(args.output_path, index=False)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
