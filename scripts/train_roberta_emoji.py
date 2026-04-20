"""
RoBERTa + Emoji Tokenization for Misogyny Detection
Train on Hatemoji + EDOS, Test on Online Misogyny
"""

import argparse
import re
from pathlib import Path

import emoji
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hatemoji_dir", type=Path,
        default=Path("datasets/Hatemoji/HatemojiBuild"))
    parser.add_argument("--edos_path", type=Path,
        default=Path("datasets/edos/data/edos_labelled_aggregated.csv"))
    parser.add_argument("--test_path", type=Path,
        default=Path("datasets/online-misogyny-eacl2021-main/data/post_ocr_dataset.csv"))
    parser.add_argument("--output_dir", type=Path,
        default=Path("results/roberta_emoji"))
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Emoji Preprocessing ───────────────────────────────────────────────────────
def process_emojis(text: str, mode: str = "demojize") -> str:
    if not isinstance(text, str):
        return ""
    if mode == "demojize":
        return emoji.demojize(text, delimiters=(" :", ": "))
    elif mode == "remove":
        return emoji.replace_emoji(text, replace="")
    return text


def clean_text(text: str, emoji_mode: str = "demojize") -> str:
    if not isinstance(text, str):
        return ""
    text = process_emojis(text, mode=emoji_mode)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Dataset ───────────────────────────────────────────────────────────────────
class MisogynyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_train_val_data(hatemoji_dir: Path, edos_path: Path, emoji_mode: str = "demojize"):
    """Load Hatemoji + EDOS for training and validation."""

    # ── Hatemoji ──
    hatemoji_dfs = []
    for split_file in ['train.csv', 'validation.csv']:
        hf = pd.read_csv(hatemoji_dir / split_file, keep_default_na=False)
        hatemoji_dfs.append(hf)
    hatemoji_df = pd.concat(hatemoji_dfs, ignore_index=True)
    hatemoji_df = hatemoji_df[['text', 'label_gold']].copy()
    hatemoji_df = hatemoji_df.rename(columns={'label_gold': 'binary_label'})
    hatemoji_df['source'] = 'hatemoji'
    print(f"Hatemoji train+val: {len(hatemoji_df)} samples ({hatemoji_df['binary_label'].sum()} hate)")

    # ── EDOS ──
    edos_df = pd.read_csv(edos_path, keep_default_na=False)
    edos_train = edos_df[edos_df['split'].isin(['train', 'dev'])].copy()
    edos_train['binary_label'] = (edos_train['label_sexist'] == 'sexist').astype(int)
    edos_train = edos_train[['text', 'binary_label']].copy()
    edos_train['source'] = 'edos'
    print(f"EDOS train+dev: {len(edos_train)} samples ({edos_train['binary_label'].sum()} sexist)")

    # ── Merge ──
    df = pd.concat([hatemoji_df, edos_train], ignore_index=True)
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x, emoji_mode=emoji_mode))
    df = df[df['cleaned_text'].str.strip() != ''].copy()

    print(f"\nTotal train+val: {len(df)} samples")
    print(f"Positive (hate/sexist): {df['binary_label'].sum()} ({df['binary_label'].mean()*100:.1f}%)")

    # Split 80/20 for train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['binary_label']
    )
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df


def load_test_data(test_path: Path, emoji_mode: str = "demojize"):
    """Load Online Misogyny as held-out test set."""
    df = pd.read_csv(test_path, keep_default_na=False)
    df = df[df['label'].isin(['Misogynistic', 'Nonmisogynistic'])].copy()
    df['binary_label'] = (df['label'] == 'Misogynistic').astype(int)
    df['cleaned_text'] = df['raw_post_text'].apply(lambda x: clean_text(x, emoji_mode=emoji_mode))
    df = df[df['cleaned_text'].str.strip() != ''].copy()
    print(f"\nOnline Misogyny test: {len(df)} samples ({df['binary_label'].sum()} misogynistic, {df['binary_label'].mean()*100:.1f}%)")
    return df


# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, scheduler, device, weights=None):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds,
        target_names=["Nonmisogynistic", "Misogynistic"]
    )
    return avg_loss, f1, report


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, val_df = load_train_val_data(
        args.hatemoji_dir, args.edos_path, emoji_mode="demojize"
    )
    test_df = load_test_data(args.test_path, emoji_mode="demojize")

    # Tokenizer & model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    # Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_df['binary_label'].values
    )
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights}")

    # Dataloaders
    train_dataset = MisogynyDataset(
        train_df["cleaned_text"].tolist(),
        train_df["binary_label"].tolist(),
        tokenizer, args.max_len
    )
    val_dataset = MisogynyDataset(
        val_df["cleaned_text"].tolist(),
        val_df["binary_label"].tolist(),
        tokenizer, args.max_len
    )
    test_dataset = MisogynyDataset(
        test_df["cleaned_text"].tolist(),
        test_df["binary_label"].tolist(),
        tokenizer, args.max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0
    for epoch in range(args.epochs):
        print(f"\n── Epoch {epoch+1}/{args.epochs} ──")

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device, weights=weights
        )
        val_loss, val_f1, val_report = evaluate(model, val_loader, device)

        print(f"Train loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")
        print(f"\nValidation Report:\n{val_report}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = args.output_dir / "best_model"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"✓ New best model saved (F1: {best_val_f1:.4f})")

    # Final test on Online Misogyny
    print("\n── Final Test on Online Misogyny ──")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.output_dir / "best_model"
    ).to(device)
    test_loss, test_f1, test_report = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f} | Test F1: {test_f1:.4f}")
    print(f"\nTest Report:\n{test_report}")

    # Save results
    with open(args.output_dir / "results.txt", "w") as f:
        f.write(f"best_val_f1: {best_val_f1}\n")
        f.write(f"test_f1: {test_f1}\n")
        f.write(f"test_report:\n{test_report}\n")
    print(f"\nResults saved to {args.output_dir}/results.txt")


if __name__ == "__main__":
    main()