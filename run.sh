#!/bin/bash
#SBATCH --job-name="DL-MAI EMBEDDINGS"
#SBATCH --partition="projects"
#SBATCH --workdir=.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128000

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

#!/bin/bash
source initialize.sh
python train.py fcn8 --epochs=200 --batch-size=64 --dataset-dir=$TMPDIR
