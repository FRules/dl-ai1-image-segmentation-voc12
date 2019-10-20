#!/bin/bash
#SBATCH --job-name="DL-MAI CNN"
#SBATCH --partition="debug"
#SBATCH --workdir=.
#SBATCH --nodes=1
#SBATCH --tasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres gpu:4
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=00:30:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

export CUDA_VISIBLE_DEVICES=0 && python3 train.py experiment_5 --epochs=1 &
export CUDA_VISIBLE_DEVICES=1 && python3 train.py experiment_6 --epochs=1 &
export CUDA_VISIBLE_DEVICES=2 && python3 train.py experiment_7 --epochs=1 &
#export CUDA_VISIBLE_DEVICES=3 && python3 train.py experiment_8 --epochs=1 &

wait
