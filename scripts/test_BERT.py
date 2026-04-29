from pathlib import Path
import json
from datetime import datetime

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from transformers import TrainingArguments


# ================= CONFIG =================
RESULTS_DIR = Path("/scratch/results")
TEST_CSV = Path("/scratch/EE559-DeepLearningProject-g37/datasets/processed/test.csv")
OUTPUT_TXT = RESULTS_DIR / "test_results.txt"
OUTPUT_JSON = RESULTS_DIR / "test_results.json"


# ================= CHECKPOINT =================
def latest_checkpoint(results_dir: Path) -> Path:
    checkpoints = sorted(
        (
            path for path in results_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.split("-")[-1].isdigit()
        ),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found in {results_dir}. Train the model first."
        )
    return checkpoints[-1]


# ================= METRICS =================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# ================= MAIN =================
print("=== TEST BERT MODEL ===")

#checkpoint_dir = latest_checkpoint(RESULTS_DIR)
checkpoint_dir = Path("/scratch/results/checkpoint-500")  # exemple
print(f"Using checkpoint: {checkpoint_dir}")

# Load data
test_df = pd.read_csv(TEST_CSV)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer + model (GPU auto)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

# Tokenization
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Trainer
training_args = TrainingArguments(
    output_dir="/scratch/tmp_trainer",
    per_device_eval_batch_size=32,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
)

# ================= EVALUATION =================
metrics = trainer.evaluate(test_dataset)

# Predictions for full report
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(axis=1)
labels = predictions.label_ids

report = classification_report(
    labels, preds,
    target_names=["Nonmisogynistic", "Misogynistic"]
)

# ================= PRINT =================
print("\n=== FINAL TEST RESULTS ===")
print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"Macro F1: {metrics['eval_f1_macro']:.4f}")

print("\nClassification Report:")
print(report)

# ================= SAVE =================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# TXT (lisible)
with open(OUTPUT_TXT, "w") as f:
    f.write("=== TEST BERT RESULTS ===\n\n")
    f.write(f"Checkpoint: {checkpoint_dir}\n\n")
    f.write(f"Accuracy: {metrics['eval_accuracy']:.4f}\n")
    f.write(f"Macro F1: {metrics['eval_f1_macro']:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# JSON (propre)
summary = {
    "timestamp": datetime.now().isoformat(),
    "checkpoint": str(checkpoint_dir),
    "accuracy": metrics["eval_accuracy"],
    "f1_macro": metrics["eval_f1_macro"],
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*50)
print(f"Results saved to:")
print(f"- {OUTPUT_TXT}")
print(f"- {OUTPUT_JSON}")
print("="*50)