#!/bin/bash

#SBATCH -N 1
#SBATCH -p GPU-small
#SBATCH -t 8:00:00
#SBATCH --gpus=v100-32:8

set -x
source $HOME/.bash_profile
cd /ocean/projects/eng170006p/vpuri1/GeomLearning.py
module load cuda anaconda3
conda active GeomLearning

# torchrun --nproc-per-node gpu -m am
./scripts/bb.sh

exit

# # submission commands
# # partitions: GPU, GPU-shared, GPU-small
# H=${1}
# N=${2}
# interact -t ${H}:00:00 --gres=gpu:v100-32:${N} -N 1 -p GPU-shared # OR
# salloc -J Interact --gres=gpu:v100-32:$(N) --nodes=1 --time=$(H):00:00 --partition=GPU-shared

#
