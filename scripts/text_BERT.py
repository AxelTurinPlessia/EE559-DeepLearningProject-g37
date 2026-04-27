import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.utils import resample, compute_class_weight
from torch.nn import CrossEntropyLoss

# Handle class imbalance with oversampling and undersampling
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


# Class imbalance handling strategy
strategy = "weights"  # options: "none", "weights", "oversample", "undersample"


# load datasets
train_df = pd.read_csv("/scratch/EE559-DeepLearningProject-g37/datasets/processed/train.csv")
val_df = pd.read_csv("/scratch/EE559-DeepLearningProject-g37/datasets/processed/val.csv")

# Apply class imbalance strategy
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

# Create Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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


# BERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

# Custom Trainer to handle class weights
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

# Training
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="logs",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# train and evaluate
trainer.train()
trainer.evaluate()

