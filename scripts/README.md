# Data Scripts

## Load External Text Datasets

Download the public CSV files from EDOS, Online Misogyny EACL 2021, and
Hatemoji, then write a normalized combined text-classification table:

```bash
python scripts/load_external_datasets.py --print-summary
```

By default this caches source CSVs under:

```text
datasets/
```

and writes:

```text
datasets/combined_external_datasets.csv
```

The combined CSV uses a shared schema:

```text
dataset,source_repo,source_url,source_file,sample_id,split,text,binary_label,label_text,task
```

Source-specific columns are retained when available. For Online Misogyny,
`image_files` records expected image filenames if you have the image directory
locally; the loader downloads CSV files only.

Useful options:

```bash
# Use already-downloaded source CSVs only.
python scripts/load_external_datasets.py --no-download

# Load only the datasets used by the current RoBERTa training recipe.
python scripts/load_external_datasets.py \
  --datasets edos online_misogyny hatemoji_build \
  --output datasets/roberta_emoji_sources.csv

# Keep ambiguous rows such as conflicting Online Misogyny labels.
python scripts/load_external_datasets.py --keep-unlabeled
```

## Regenerate OCR Results

Run EasyOCR on the dataset image directory and save the OCR cache:

```bash
python scripts/generate_ocr_results.py
```

By default, the script scans every image file in:

```text
datasets/online-misogyny-eacl2021-main/data/dataset_post_images
```

and writes:

```text
.easyocr/ocr_results.csv
```

Useful options:

```bash
# Recompute OCR instead of reusing valid cached rows.
python scripts/generate_ocr_results.py --refresh

# OCR only image files that are expected from final_labels.csv.
python scripts/generate_ocr_results.py --selection expected-files

# Quick smoke test without overwriting the main OCR cache.
python scripts/generate_ocr_results.py \
  --refresh \
  --limit 1 \
  --cache-input .easyocr/ocr_results.csv \
  --output /tmp/ocr_easyocr_smoke.csv
```

## Merge OCR With Labels/Text

Create the compact one-row-per-post dataset:

```bash
python scripts/merge_ocr_with_labels.py
```

This reads:

```text
datasets/online-misogyny-eacl2021-main/data/final_labels.csv
.easyocr/ocr_results.csv
```

and writes:

```text
datasets/post_ocr_dataset.csv
```

The output columns are:

```text
post_id,label,raw_post_text,image_exists,ocr_text,ocr_readable
```
