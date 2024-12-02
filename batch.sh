#!/bin/bash

H=${1}
N=${2}

# interact -t ${H}:00:00 --gres=gpu:v100-32:${N} -N 1 -p GPU
# salloc -J Interact --gres=

# interact -t ${H}:00:00 --gres=gpu:v100-32:${N} -p GPU-shared
# salloc -J Interact --gres=gpu:v100-32:$(N) --time=$(H):00:00 --partition=GPU-shared

# #SBATCH -N 1
# #SBATCH -p GPU
# #SBATCH -t 5:00:00
# #SBATCH --gpus=v100-32:8

set -x
cd /ocean/projects/eng170006p/vpuri1/GeomLearning.py
module load cuda anaconda3
conda activate GeomLearning
torchrun --nproc-per-node gpu -m am

exit
#
