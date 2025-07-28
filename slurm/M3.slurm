#!/bin/bash
#SBATCH --job-name=M3_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=[0,8-10]
#SBATCH --nodelist=gruenau7,gruenau8
#SBATCH --output=logs/M3_%a.out
#SBATCH --error=logs/M3_%a.err


source ~/.bashrc
conda activate env

cd "$(dirname "$0")/../src/nlp"
echo "Current directory: $(pwd)"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # Run the full case dataset only once
    export M3_MODE=case

    python M0_UMLS.py
    python M3.py
else
    # Run 10 chunks of the control dataset
    export M3_MODE=control
    export NUM_CHUNKS=10

    python M0_UMLS.py
    python M3.py
fi