#!/bin/bash
#SBATCH --job-name=M1              # or longgpu     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00              # adjust as needed
#SBATCH --exclusive
#SBATCH --output=logs/M1.out
#SBATCH --error=logs/M1.err

# Activate your environment
source ~/.bashrc
conda activate env

cd "$(dirname "$0")/../src/nlp"
echo "Current directory: $(pwd)"

echo "Running M0_preprocessing.py"
python M0_preprocessing.py

echo "Running M1.py"
python M1.py
