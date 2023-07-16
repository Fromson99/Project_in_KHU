#!/bin/bash

#SBATCH --job-name slurm-tutorial
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH -o slurm/logs/slurm-%A-%x.out

python competition_1.py

exit 0