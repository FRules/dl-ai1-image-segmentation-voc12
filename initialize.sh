#!/bin/bash
module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML
if [ ! -d "$TMPDIR/VOCdevkit Training" ]; then
  cp -R VOCdevkit\ Training $TMPDIR
fi
nvidia-smi