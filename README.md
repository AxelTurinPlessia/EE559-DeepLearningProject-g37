# EE559 Deep Learning Project - Group 37

## Project overview
This repository contains the code, experiments, and supporting files for our EE559 Deep Learning group project.

Our goal is to develop, train, and evaluate deep learning models for misogyny detection in texts and images (memes).

## Team
- Léontine Lefranc
- Clément Meddeb
- Axel Turin-Plessia

## Project Deliverables
- [Final Report](./Report_group37.pdf)
- [Project Poster](./Poster_345240_346164_362559.pdf)

## Repository structure
```text
EE559-DeepLearningProject-g37/
├── README.md
├── main.py              # final demo/result summary entrypoint
├── requirements.txt
├── .gitignore
├── scripts/             # data preparation and analysis scripts
├── scripts/training/    # training and fine-tuning scripts
├── datasets/            # local datasets; most are ignored by git
└── results/             # metrics, predictions, and model checkpoints
```

## Final results demo

Run the single entrypoint used for the screencast:

```bash
python main.py
```

The script prints the checkpoint inventory, summarizes the stored RoBERTa
metrics used in the report, shows the analysis tables used in the report, and
prints cached MAMI CLIP/RoBERTa+CLIP metrics. It never trains or fine-tunes a
model. By default it reads saved result files from `results/`.

The repository uses Git LFS for the large model checkpoints. After cloning,
run:

```bash
git lfs install
git lfs pull
```

Useful options:

```bash
# Only show stored metrics and skip loading MAMI .pt checkpoints.
python main.py --skip-mami-eval

# Recompute MAMI held-out evaluation metrics from saved checkpoints.
python main.py --refresh-mami-metrics
```

Training and fine-tuning scripts are in `scripts/training/`. Data preparation
and analysis scripts remain in `scripts/`.

## Reproducibility

There are three levels of reproduction.

### 1. Show the final report results

This is the command intended for the screencast and for a quick correction run:

```bash
python main.py
```

It expects the final `results/` artifacts to be present. It reads stored
metrics, predictions, and cached MAMI metrics. It does not retrain any model.
It only does light aggregation from saved prediction CSVs, such as computing
accuracy or a false-positive rate from already saved predictions.

### 2. Recompute metrics from saved checkpoints

To recompute MAMI metrics from the saved `.pt` checkpoints:

```bash
python main.py --refresh-mami-metrics
```

This loads the saved MAMI checkpoints and runs inference on
`datasets/MAMI/validation.tsv`, which is the held-out evaluation split used by
the MAMI training scripts. It does not train models.

To recompute the post/OCR error and bias analysis from saved predictions:

```bash
python scripts/analysis_step4.py \
  --predictions_path results/roberta_mixed_post_ocr/test_predictions.csv \
  --output_dir results/analysis_step4_roberta_mixed_post_ocr
```

To recompute the synthetic diagnostic benchmark for a saved RoBERTa checkpoint:

```bash
python scripts/step4_synthetic_analysis.py \
  --model_path results/roberta_mixed_post_ocr/best_model \
  --output_dir results/synthetic_diagnostics
```

### 3. Retrain or fine-tune models

Training scripts are kept in `scripts/training/`. These scripts are not called
by `main.py`; they are provided for full reproduction and usually require a GPU
or a Run:ai job.

Examples:

```bash
python scripts/training/roberta_base.py
python scripts/training/roberta_emoji.py
python scripts/training/roberta_mixed.py
python scripts/training/roberta_synthetic.py
python scripts/training/clip_mami_step5.py
python scripts/training/multimodal_roberta_base_step5.py
python scripts/training/multimodal_roberta_synthetic_step5.py
python scripts/training/multimodal_roberta_FT_step5.py
```

The stored `results/roberta_mixed_synthetic` checkpoint was trained with a
20% synthetic share and `synthetic_sampling=auto`; because the synthetic pool
contains 2,000 rows, that run reuses synthetic examples. If the intended setup
is "synthetic examples passed once" (about 8% synthetic share), retrain
`roberta_synthetic.py` with a without-replacement setup/output directory and
then retrain the corresponding `roberta_synthetic + CLIP` model.

## Datasets

The final generated text datasets and synthetic diagnostic datasets are tracked
in this repository:

```text
datasets/combined_external_datasets.csv
datasets/post_ocr_dataset.csv
datasets/synthetic/
```

The other text datasets can be loaded with:

```bash
python scripts/load_external_datasets.py --print-summary
python scripts/generate_ocr_results.py
python scripts/merge_ocr_with_labels.py
python scripts/create_mixed_roberta_dataset.py
```

What these commands do:

- `load_external_datasets.py` downloads/normalizes EDOS, Online Misogyny CSV
  labels, and Hatemoji CSV files under `datasets/`.
- `generate_ocr_results.py` runs EasyOCR on Online Misogyny images and writes
  `.easyocr/ocr_results.csv`.
- `merge_ocr_with_labels.py` combines `final_labels.csv` with OCR text and
  writes `datasets/post_ocr_dataset.csv`.
- `create_mixed_roberta_dataset.py` creates the mixed RoBERTa train/validation
  and held-out post/OCR test splits under `datasets/roberta_mixed_post_ocr/`.

The MAMI image dataset is not tracked in this repository. Download it manually
from the shared Google Drive folder:

```text
https://drive.google.com/drive/folders/1x04eqdhH_JBadUeutIf02szK_778mmHH?usp=sharing
```

If the downloaded archive asks for a password, use:

```text
*MaMiSemEval2022!
```

Then extract or move the dataset so that it is placed at:

```text
datasets/MAMI/
├── train.tsv
├── validation.tsv
├── test.tsv
└── MAMI_2022_images/
    ├── training_images/
    └── test_images/
```

In other words, after installation the repository root should contain
`datasets/MAMI/train.tsv`, `datasets/MAMI/validation.tsv`, and
`datasets/MAMI/MAMI_2022_images/`.

Current local dataset sizes:

```text
datasets/Hatemoji/                         ~842 KB
datasets/edos/                             ~3.7 MB
datasets/combined_external_datasets.csv    ~13 MB
datasets/online-misogyny-eacl2021-main/    ~3.3 MB
datasets/post_ocr_dataset.csv              ~2.1 MB
datasets/roberta_mixed_post_ocr/           ~16 MB
datasets/synthetic/                        ~746 KB
datasets/MAMI/                             ~1.8 GB, manually installed
```

The raw external dataset folders can be regenerated with the scripts above. The
MAMI folder should not be pushed to Git because it is about 1.8 GB and contains
many image files. It is ignored by `.gitignore` and should be installed
manually from the Drive link above.

## Results and model artifacts

`main.py` expects the final `results/` artifacts to exist. These include both
small metric files and large model checkpoints:

```text
results/roberta_base/
results/roberta_emoji/
results/roberta_mixed_post_ocr/
results/roberta_mixed_synthetic/
results/results_clip_mami/
results/results_multimodal_roberta_base/
results/results_multimodal_roberta_synthetic/
results/results_multimodal_weighted/
results/analysis_step4_roberta_mixed_post_ocr/
results/synthetic_diagnostics/
```

Do not remove `results/` from `.gitignore` and push everything blindly. Some
checkpoint files are hundreds of MB or more than 1 GB, so GitHub requires Git
LFS for them. This repository tracks only the final selected result folders and
uses Git LFS for:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.safetensors"
git add .gitattributes
```

Then explicitly unignore/add only the final result folders above. Old BERT
outputs, smoke-test outputs, `results/results_full/`, and
`results/roberta_mami_step5/` are not needed for the final demo.

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
