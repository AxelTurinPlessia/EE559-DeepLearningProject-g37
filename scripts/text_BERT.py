import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
from sklearn.utils import resample, compute_class_weight
from torch.nn import CrossEntropyLoss

# ===============================
# CONFIG
# ===============================

strategy = "undersample"  # "none", "weights", "oversample", "undersample"

MODEL_PATH = os.environ.get("MODEL_PATH", "/scratch/bert_model")

DATA_PATH = "/scratch/EE559-DeepLearningProject-g37/datasets/processed"

# ===============================
# CLASS IMBALANCE
# ===============================

def oversample_df(df):
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    return df_balanced.sample(frac=1).reset_index(drop=True)


def undersample_df(df):
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    return df_balanced.sample(frac=1).reset_index(drop=True)


# ===============================
# LOAD DATA
# ===============================

train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
val_df = pd.read_csv(f"{DATA_PATH}/val.csv")

if strategy == "oversample":
    train_df = oversample_df(train_df)
elif strategy == "undersample":
    train_df = undersample_df(train_df)

if strategy == "weights":
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values
    )
    weights = torch.tensor(class_weights, dtype=torch.float)
else:
    weights = None

# ===============================
# DATASETS
# ===============================

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ===============================
# TOKENIZER
# ===============================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ===============================
# MODEL
# ===============================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    local_files_only=True
)

# ===============================
# METRICS
# ===============================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

# ===============================
# CUSTOM TRAINER
# ===============================

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if weights is not None:
            loss_fct = CrossEntropyLoss(weight=weights.to(logits.device))
        else:
            loss_fct = CrossEntropyLoss()

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ===============================
# TRAINING
# ===============================

training_args = TrainingArguments(
    output_dir="/scratch/results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="/scratch/logs",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ===============================
# TRAIN
# ===============================

trainer.train()

# ===============================
# FINAL EVALUATION
# ===============================

print("\n── Final Evaluation ──")

predictions = trainer.predict(val_dataset)

logits = predictions.predictions
labels = predictions.label_ids
preds = logits.argmax(axis=1)

accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="macro")

report = classification_report(
    labels,
    preds,
    target_names=["Nonmisogynistic", "Misogynistic"]
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {f1:.4f}")
print("\nClassification Report:\n", report)

# ===============================
# SAVE RESULTS
# ===============================

output_path = f"/scratch/results/text_bert_{strategy}.txt"

os.makedirs("/scratch/results", exist_ok=True)

with open(output_path, "w") as f:
    f.write("=== TEXT BERT RESULTS ===\n\n")
    f.write(f"Strategy: {strategy}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"\nResults saved to {output_path}")