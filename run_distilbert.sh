#!/bin/bash
#SBATCH --job-name=iberlef_distilbert
#SBATCH --output=logs/distilbert_%j.out
#SBATCH --error=logs/distilbert_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
module load conda
source activate ~/envs/iberlef

cd ~/IberLEF_Challenge
mkdir -p logs models

python train_distilbert.py
