#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=smoke_%j.out

python3 smoke_test.py