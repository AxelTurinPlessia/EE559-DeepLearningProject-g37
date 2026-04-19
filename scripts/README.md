# Data Scripts

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
datasets/online-misogyny-eacl2021-main/data/post_ocr_dataset.csv
```

The output columns are:

```text
post_id,label,raw_post_text,image_exists,ocr_text,ocr_readable
```
