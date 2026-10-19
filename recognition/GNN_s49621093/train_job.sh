#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=gnn_%j.out

source /home/Student/s4962109/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd ~/PatternAnalysis-2026/recognition/GNN_s49621093
python train.py --epochs 200 --out outputs/seed0