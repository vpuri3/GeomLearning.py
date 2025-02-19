#!/bin/bash

#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH -p GPU
#SBATCH --gpus=v100-32:8
#SBATCH --reservation=GPUeng170006p
#SBATCH --output=slurm-%j.out  # Default output file

set -x
source $HOME/.bash_profile
cd /ocean/projects/eng170006p/vpuri1/GeomLearning.py
module load cuda anaconda3
conda activate GeomLearning

EXP_NAME="tra_steady_sdf_width_192_slices_064_wd_0p1"

torchrun \
    --nproc-per-node gpu \
    -m am \
    --exp_name ${EXP_NAME} \
    --train true \
    --epochs 500 \
    --timeseries false \
    --sdf true \
    --TRA true \
    --tra_width 192 \
    --tra_num_slices 64 \
    --weight_decay 1e-1

cp slurm-${SLURM_JOB_ID}.out out/am/${EXP_NAME}/

exit

# # submission commands
# # partitions: GPU, GPU-shared, GPU-small
# H=${1}
# N=${2}
# interact -t ${H}:00:00 --gres=gpu:v100-32:${N} -N 1 -p GPU-shared # OR
# salloc -J Interact --gres=gpu:v100-32:$(N) --nodes=1 --time=$(H):00:00 --partition=GPU-shared
#