# EE559 Deep Learning Project — Group 37
test1
test2
test3
## Project overview
This repository contains the code, experiments, and supporting files for our EE559 Deep Learning group project.

Our goal is to develop, train, and evaluate deep learning models for misogyny detection in texts and images (memes).

## Team
- Léontine Lefranc
- Clément Meddeb
- Axel Turin-Plessia


## Repository structure
```text
EE559-DeepLearningProject-g37/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                 # main source code
├── notebooks/           # exploratory notebooks
├── scripts/             # training / evaluation scripts
├── configs/             # configuration files
├── data/                # local data pointers or small metadata files only
├── results/             # generated results (ignored by git if large)
└── checkpoints/         # trained model weights (ignored by git if large)

## Synthetic RoBERTa Fine-Tuning

To fine-tune the current best RoBERTa checkpoint with synthetic training-only
augmentation, run:

```bash
python scripts/09_finetune_roberta_on_synthetic.py \
  --synthetic_train_path datasets/synthetic/synthetic_train_set.jsonl \
  --base_model_path results/roberta_mixed_post_ocr/lr2e-5_seed42/best_model \
  --output_dir results/roberta_synthetic_finetune \
  --epochs 4 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --seed 42
```

The script trains only on `datasets/synthetic/synthetic_train_set.jsonl`. It
uses the real validation split in `datasets/roberta_mixed_post_ocr/val.csv` for
checkpoint selection, then evaluates the selected checkpoint on EDOS test,
HatemojiBuild test, and the held-out Online Misogyny post/OCR test split. It
also evaluates the synthetic diagnostic benchmark files in `datasets/synthetic/`.

Results are saved under `results/roberta_synthetic_finetune/`, including
separate real-world and synthetic benchmark metrics, classification reports,
confusion matrices, predictions, and the new checkpoint. Synthetic data is an
augmentation source only; it is not a substitute for real-world held-out
evaluation.
