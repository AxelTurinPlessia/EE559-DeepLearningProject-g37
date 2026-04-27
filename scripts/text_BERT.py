import json
from datetime import datetime
from pathlib import Path

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

# load datasets
train_df = pd.read_csv("datasets/processed/train.csv")
val_df = pd.read_csv("datasets/processed/val.csv")

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
        "f1": f1_score(labels, preds)
    }

# Training
training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# train and evaluate
print("\n" + "="*50)
print("TRAINING IN PROGRESS...")
print("="*50)
trainer.train()

print("\n" + "="*50)
print("EVALUATING ON VALIDATION SET...")
print("="*50)
eval_results = trainer.evaluate()

# Save results to file
results_file = Path("results") / "training_results.json"
results_file.parent.mkdir(parents=True, exist_ok=True)

results_summary = {
    "timestamp": datetime.now().isoformat(),
    "model": "bert-base-uncased",
    "training_epochs": 2,
    "batch_size": 16,
    "max_length": 128,
    "validation_results": eval_results
}

with open(results_file, "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "="*50)
print(f"Results saved to: {results_file}")
print("="*50)