#!/bin/bash
#SBATCH --job-name="DL-MAI CNN"
#SBATCH --partition="projects"
#SBATCH --workdir=.
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=2:00:00
#SBATCH --exclusive

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

srun -n1 -N1 python3 train.py experiment_1 --epochs=35 --batch-size=64 &
srun -n1 -N1 python3 train.py experiment_4 --epochs=35 --batch-size=64 &
srun -n1 -N1 python3 train.py experiment_5 --epochs=35 --batch-size=64 &
srun -n1 -N1 python3 train.py fcn8 --epochs=25 --batch-size=64 --weights=models/fcn8/vgg16_weights.h5 &
wait
