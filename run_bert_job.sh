#!/bin/bash
#SBATCH --job-name=bert_training
#SBATCH --output=logs/bert_training_%j.out
#SBATCH --error=logs/bert_training_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load modules if needed (uncomment and modify as needed)
# module load python/3.10
# module load cuda/11.8

# Activate environment if you have one
# conda activate dl_project

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/home/meddeb/.local/lib/python3.10/site-packages

# Create directories
mkdir -p logs results

# Run the training script
python text_BERT.py

# Run the test script
python test_BERT.py