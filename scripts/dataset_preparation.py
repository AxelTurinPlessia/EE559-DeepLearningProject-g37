import pandas as pd
from sklearn.model_selection import train_test_split
import os

# load dataset
df = pd.read_csv("datasets/edos/data/edos_labelled_aggregated.csv")

# we only keep the sexism text and clean the labels
df = df[["text", "label_sexist"]]
df = df.rename(columns={"label_sexist": "label"})
df["label"] = df["label"].apply(lambda x: 1 if x == "sexist" else 0)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# check distribution
print(df.head())
print(df["label"].value_counts())

# split dataset into train, val, test (80-10-10) with stratification
train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

# save datasets
os.makedirs("datasets/processed", exist_ok=True)

train.to_csv("datasets/processed/train.csv", index=False)
val.to_csv("datasets/processed/val.csv", index=False)
test.to_csv("datasets/processed/test.csv", index=False)

print("Datasets saved to datasets/processed/")
