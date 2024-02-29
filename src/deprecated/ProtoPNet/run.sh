#!/usr/bin/env bash

#SBATCH --job-name=protopnet_cub      # Job name
#SBATCH --output=logs/my_job_log_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                  # Job memory request
#SBATCH --time=10:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:4

eval "$(conda shell.bash hook)" 
conda activate protopnet
python3 ProtoPNet/main.py