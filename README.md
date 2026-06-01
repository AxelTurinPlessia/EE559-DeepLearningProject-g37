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

## Final results demo

Run the single entrypoint used for the screencast:

```bash
python main.py
```

The script prints the checkpoint inventory, summarizes the stored RoBERTa
metrics used in the report, and evaluates the saved MAMI CLIP/RoBERTa+CLIP
checkpoints when cached MAMI metrics are missing. It never trains or
fine-tunes a model. After the first MAMI evaluation, metrics are cached under
the corresponding `results/*/metrics_validation.json` files so later runs are
fast.

Useful options:

```bash
# Only show stored metrics and skip loading MAMI .pt checkpoints.
python main.py --skip-mami-eval

# Recompute MAMI validation metrics from saved checkpoints.
python main.py --refresh-mami-metrics
```

Training and fine-tuning scripts are in `scripts/training/`. Data preparation
and analysis scripts remain in `scripts/`.

## Mixed Real/Synthetic RoBERTa Fine-Tuning

The mixed real/synthetic experiment keeps the model grounded in the original
mixed real training split while adding synthetic examples as a smaller
augmentation source.

Run the 80/20 real/synthetic experiment with:

```bash
python scripts/training/roberta_synthetic.py \
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
