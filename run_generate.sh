#!/bin/bash
#SBATCH --job-name=iberlef_generate
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
module load conda
module load rocm/6.4.3
source activate ~/envs/iberlef

cd ~/IberLEF_Challenge
mkdir -p logs

python generate_data.py

