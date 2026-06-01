import os
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils import resample, compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ================= CONFIG =================
BASE_PATH = Path("/scratch/EE559-DeepLearningProject-g37")
DATA_PATH = BASE_PATH / "datasets/processed"

OUTPUT_DIR = Path("/scratch/results_full")
MODEL_SAVE_PATH = OUTPUT_DIR / "best_model"

TRAIN_FILE = DATA_PATH / "train.csv"
VAL_FILE = DATA_PATH / "val.csv"
TEST_FILE = DATA_PATH / "test.csv"

STRATEGY = os.environ.get("STRATEGY", "none")  # none / weights / oversample / undersample

# ================= IMBALANCE =================
def oversample_df(df):
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    return pd.concat([df_majority, df_minority_upsampled]).sample(frac=1)

def undersample_df(df):
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )
    return pd.concat([df_majority_downsampled, df_minority]).sample(frac=1)

# ================= LOAD DATA =================
train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

# Apply strategy
if STRATEGY == "oversample":
    train_df = oversample_df(train_df)

elif STRATEGY == "undersample":
    train_df = undersample_df(train_df)

if STRATEGY == "weights":
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values
    )
    weights = torch.tensor(class_weights, dtype=torch.float)
else:
    weights = None

# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ================= MODEL =================
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# ================= METRICS =================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

# ================= TRAINER =================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="/scratch/tmp_trainer",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ================= TRAIN =================
print(f"\n=== TRAINING (strategy={STRATEGY}) ===")
trainer.train()

# ================= VALIDATION =================
val_metrics = trainer.evaluate()
print("\n=== VALIDATION ===")
print(val_metrics)

# Save best model
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

# ================= TEST =================
print("\n=== TEST ===")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)

metrics = trainer.evaluate(test_dataset)

predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(axis=1)
labels = predictions.label_ids

report = classification_report(
    labels, preds,
    target_names=["Nonmisogynistic", "Misogynistic"]
)

print("\n=== FINAL TEST RESULTS ===")
print(metrics)
print(report)

# ================= SAVE RESULTS =================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TRAIN / VAL
train_file = OUTPUT_DIR / f"train_results_{STRATEGY}.txt"
with open(train_file, "w") as f:
    f.write(f"Strategy: {STRATEGY}\n")
    f.write(f"Validation metrics:\n{val_metrics}\n")

# TEST
test_file = OUTPUT_DIR / f"test_results_{STRATEGY}.txt"
with open(test_file, "w") as f:
    f.write(f"Strategy: {STRATEGY}\n")
    f.write(f"Test metrics:\n{metrics}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# JSON
json_file = OUTPUT_DIR / f"summary_{STRATEGY}.json"
summary = {
    "strategy": STRATEGY,
    "timestamp": datetime.now().isoformat(),
    "val": val_metrics,
    "test": metrics
}
with open(json_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved in {OUTPUT_DIR}")