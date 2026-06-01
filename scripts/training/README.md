# Training and fine-tuning scripts

These scripts create or fine-tune model checkpoints. They are kept out of the
main `scripts/` folder because the final demo entrypoint is `main.py`, which
summarizes stored results and evaluates saved checkpoints without retraining.

The scripts in this folder are not called by `main.py`.
