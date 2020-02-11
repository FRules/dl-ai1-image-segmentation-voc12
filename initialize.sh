#!/bin/bash
module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML
if [[ $@ == *"dataset"* ]]; then
  echo "Dataset maybe need to be copied"
  if [ ! -d "$TMPDIR/VOCdevkit Training" ]; then
    echo "Copying dataset to $TMPDIR"
    cp -R VOCdevkit\ Training $TMPDIR
  fi
fi

nvidia-smi