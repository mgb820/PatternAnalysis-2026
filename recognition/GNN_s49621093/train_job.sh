#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=gnn_%j.out

cd ~/PatternAnalysis-2026/recognition/GNN_s49621093
python python train.py --epochs 200 --out outputs/seed0