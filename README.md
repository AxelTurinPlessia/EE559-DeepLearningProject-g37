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
```

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

## Mixed Real/Synthetic RoBERTa Fine-Tuning

The synthetic-only continued fine-tune improves synthetic diagnostics but
over-predicts misogyny on real-world data. The mixed real/synthetic experiment
keeps the model grounded in the original mixed real training split while adding
synthetic examples as a smaller augmentation source.

Run the 80/20 real/synthetic experiment with:

```bash
python scripts/10_finetune_roberta_mixed_real_synthetic.py \
  --base_model_path results/roberta_mixed_post_ocr/lr2e-5_seed42/best_model \
  --real_train_path datasets/roberta_mixed_post_ocr/train.csv \
  --real_val_path datasets/roberta_mixed_post_ocr/val.csv \
  --synthetic_train_path datasets/synthetic/synthetic_train_set.jsonl \
  --synthetic_ratio 0.20 \
  --synthetic_sampling auto \
  --output_dir results/roberta_mixed_real80_synthetic20_lr5e-6_seed42 \
  --epochs 1 \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --seed 42 \
  --tune_threshold
```

The script keeps all real training examples and samples synthetic examples so
that they represent about 20% of the final training set. If the synthetic pool
is too small, `--synthetic_sampling auto` samples with replacement. Checkpoint
selection uses only the real validation split. With `--tune_threshold`, the
decision threshold is selected on the real validation split by maximizing macro
F1 over thresholds from 0.05 to 0.95. Test data and synthetic diagnostic
benchmarks are never used for training, checkpoint selection, or threshold
tuning.

Outputs are saved under the chosen output directory, including model
checkpoints, configuration, training-data summaries, threshold metadata,
real-world and synthetic benchmark reports at threshold 0.5 and the tuned
threshold, confusion matrices, predictions, and `model_comparison.csv` when
previous result files are available.
