"""
Step 5: Multimodal Misogyny Detection (RoBERTa + CLIP)
- Uses fine-tuned RoBERTa
- Uses CLIP for image features
- Handles class imbalance with weighted loss
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from PIL import Image

from transformers import AutoTokenizer, AutoModel
import open_clip

from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRATCH_ROOT = Path("/scratch")

ROBERTA_PATH = SCRATCH_ROOT / "EE559-DeepLearningProject-g37/results/roberta_mixed_post_ocr/best_model"
DATASET_PATH = SCRATCH_ROOT / "mami_dataset"

BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 3


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class MAMIDataset(Dataset):
    def __init__(self, tsv_file, image_root, tokenizer):
        self.df = pd.read_csv(tsv_file, sep="\t")
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer

        if "label" not in self.df.columns:
            raise ValueError(f"'label' column missing: {self.df.columns}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["text"])
        image_path = self.image_root / row["file_name"]
        label = int(row["label"])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        image = Image.open(image_path).convert("RGB")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label)
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "images": [x["image"] for x in batch],
        "labels": torch.stack([x["label"] for x in batch]),
    }


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()

        # RoBERTa
        self.text_model = AutoModel.from_pretrained(ROBERTA_PATH)

        # CLIP
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )

        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Projection layers
        self.text_proj = nn.Linear(768, 256)
        self.image_proj = nn.Linear(512, 256)

        # Classifier
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, images):
        # TEXT
        text_out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_out.last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_feat)

        # IMAGE
        images = torch.stack([self.preprocess(img) for img in images]).to(DEVICE)
        image_feat = self.clip_model.encode_image(images)
        image_feat = image_feat.float()
        image_feat = self.image_proj(image_feat)

        # FUSION
        x = torch.cat([text_feat, image_feat], dim=1)
        return self.classifier(x)


# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────

def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask, batch["images"])
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits = model(input_ids, attention_mask, batch["images"])
            p = torch.argmax(logits, dim=1)

            preds.extend(p.cpu().numpy())
            labels.extend(batch["labels"].numpy())

    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)

    return f1, acc


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)

    train_dataset = MAMIDataset(
        DATASET_PATH / "train.tsv",
        DATASET_PATH / "MAMI_2022_images/training_images",
        tokenizer
    )

    val_dataset = MAMIDataset(
        DATASET_PATH / "validation.tsv",
        DATASET_PATH / "MAMI_2022_images/training_images",
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = MultimodalModel().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 🔥 CLASS IMBALANCE HANDLING
    labels = train_dataset.df["label"].values

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels
    )

    weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Class weights: {class_weights}")

    loss_fn = nn.CrossEntropyLoss(weight=weights)

    print("\n=== TRAINING MULTIMODAL (WEIGHTED) ===")

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer, loss_fn)
        f1, acc = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {loss:.4f}")
        print(f"Val F1: {f1:.4f}")
        print(f"Val Acc: {acc:.4f}")

    save_path = SCRATCH_ROOT / "results_multimodal_weighted"
    save_path.mkdir(exist_ok=True)

    torch.save(model.state_dict(), save_path / "model.pt")
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()