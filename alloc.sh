#!/bin/bash
salloc -t 02:00:00 -n 1 -c 4 --gres=gpu:1 --mem 32000 srun --pty /bin/bash