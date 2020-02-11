#!/bin/bash
#SBATCH --job-name="DL-MAI EMBEDDINGS"
#SBATCH --partition="debug"
#SBATCH --workdir=.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH --output=logs/stdout.log
#SBATCH --error=logs/stderr.log
#SBATCH --time=01:00:00
#SBATCH --mem=128000

echo "Run with params: $@"
source initialize.sh "$@"
python train.py "$@"
