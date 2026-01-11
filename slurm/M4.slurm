#!/bin/bash
#SBATCH --job-name=M4
#SBATCH --partition=gpu              # or longgpu
#SBATCH --gres=gpu:1    
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00              
#SBATCH --array=[1-11]                # 12 configurations (4 types of models and 3 horizons)
#SBATCH --output=logs/M4_%a.log
#SBATCH --error=logs/M4_%a.log

source ~/.bashrc
conda activate env_model

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python3 main.py
