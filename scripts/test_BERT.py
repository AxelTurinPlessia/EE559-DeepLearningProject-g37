from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


RESULTS_DIR = Path("results")
TEST_CSV = Path("datasets/processed/test.csv")


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


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


checkpoint_dir = latest_checkpoint(RESULTS_DIR)
print(f"Loading checkpoint: {checkpoint_dir}")

test_df = pd.read_csv(TEST_CSV)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

metrics = trainer.evaluate(test_dataset)
print("Test results:")
print(metrics)
