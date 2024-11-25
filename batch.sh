#!/bin/bash

H=${1}
N=${2}

interact -t ${H}:00:00 --gres=gpu:v100-32:${N} -p GPU-shared
## salloc -J Interact --gres=gpu:v100-32:$(N) --time=$(H):00:00 --partition=GPU-shared

set -x
cd /ocean/projects/eng170006p/vpuri1/GeomLearning.py
module load cuda anaconda3
conda activate GeomLearning

torchrun --nproc-per-node gpu -m am
exit
#
