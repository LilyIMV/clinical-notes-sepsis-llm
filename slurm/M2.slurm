#!/bin/bash
#SBATCH --job-name=M2
#SBATCH --partition=gpu              # or longgpu
#SBATCH --gres=gpu:1    
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=48:00:00              
#SBATCH --output=logs/M2_%a.out
#SBATCH --error=logs/M2_%a.err

source ~/.bashrc
conda activate env

cd "$(dirname "$0")/../src/nlp"
echo "Current directory: $(pwd)"

echo "Running M0_preprocessing.py"
python M0_preprocessing.py

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # Run the full case dataset only once
    export M2_MODE=case

    conda deactivate
    conda activate env_RAG
    echo "Running M0_RAG.py"
    python M0_RAG.py

    conda deactivate
    conda activate env
    python M2.py
else
    # Run 10 chunks of the control dataset
    export M2_MODE=control
    export NUM_CHUNKS=10
    
    conda deactivate
    conda activate env_RAG
    echo "Running M0_RAG.py"
    python M0_RAG.py

    conda deactivate
    conda activate env
    python M2.py
fi