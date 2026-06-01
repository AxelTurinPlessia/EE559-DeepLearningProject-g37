"""
Step 5: CLIP-only Misogyny Detection (Full Meme Image)
- Uses CLIP image embeddings from the full meme image
- No tokenizer, OCR, or explicit text input
- Handles class imbalance with weighted loss
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from PIL import Image

import open_clip

from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRATCH_ROOT = Path("/scratch")
DATASET_PATH = SCRATCH_ROOT / "mami_dataset"

BATCH_SIZE = 16
EPOCHS = 3


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class MAMIDataset(Dataset):
    def __init__(self, tsv_file, image_root):
        self.df = pd.read_csv(tsv_file, sep="\t")
        self.image_root = Path(image_root)

        if "label" not in self.df.columns:
            raise ValueError(f"'label' column missing: {self.df.columns}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_root / row["file_name"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "label": torch.tensor(label)
        }


def collate_fn(batch):
    return {
        "images": [x["image"] for x in batch],
        "labels": torch.stack([x["label"] for x in batch]),
    }


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class CLIPOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CLIP
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )

        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, images):
        # IMAGE
        images = torch.stack([self.preprocess(img) for img in images]).to(DEVICE)
        image_feat = self.clip_model.encode_image(images)
        image_feat = image_feat.float()

        # CLASSIFICATION
        return self.classifier(image_feat)


# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────

def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        labels = batch["labels"].to(DEVICE)

        logits = model(batch["images"])
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
            logits = model(batch["images"])
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

    train_dataset = MAMIDataset(
        DATASET_PATH / "train.tsv",
        DATASET_PATH / "MAMI_2022_images/training_images"
    )

    val_dataset = MAMIDataset(
        DATASET_PATH / "validation.tsv",
        DATASET_PATH / "MAMI_2022_images/training_images"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = CLIPOnlyModel().to(DEVICE)

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

    print("\n=== TRAINING CLIP (FULL MEME IMAGE ONLY) ===")

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer, loss_fn)
        f1, acc = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {loss:.4f}")
        print(f"Val F1: {f1:.4f}")
        print(f"Val Acc: {acc:.4f}")

    save_path = SCRATCH_ROOT / "results_clip_mami"
    save_path.mkdir(exist_ok=True)

    torch.save(model.state_dict(), save_path / "model.pt")
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
