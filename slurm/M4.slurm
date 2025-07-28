#!/bin/bash
#SBATCH --job-name=M4
#SBATCH --partition=gpu              # or longgpu
#SBATCH --gres=gpu:1    
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00              
#SBATCH --array=0-11                # 12 configurations
#SBATCH --output=logs/M4_%a.log
#SBATCH --error=logs/M4_%a.log

source ~/.bashrc
conda activate env_model

cd "$(dirname "$0")/../src/model"
echo "Current directory: $(pwd)"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python main.py
