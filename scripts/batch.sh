#!/bin/bash

#SBATCH -N 1
#SBATCH -t 16:00:00
#SBATCH -p GPU
#SBATCH --gpus=v100-32:8
#SBATCH --output=slurm-%j.out
# #SBATCH --reservation=GPUeng170006p

set -x
source $HOME/.bash_profile
cd /ocean/projects/eng170006p/vpuri1/GeomLearning.py
module load cuda anaconda3
conda activate GeomLearning

# TIMESERIES
EXP_NAME="tra_timeseries_sdf_layers_8_width_128_slices_32_wd_0p01_heads_8_sdf_false"
torchrun \
    --nproc-per-node gpu \
    -m am \
    --exp_name ${EXP_NAME} \
    --train true \
    --epochs 500 \
    --timeseries true \
    --sdf false \
    --TRA true \
    --tra_width 128 \
    --tra_num_heads 8 \
    --tra_num_slices 32 \
    --tra_num_layers 8 \
    --weight_decay 1e-2

cp slurm-${SLURM_JOB_ID}.out out/am/${EXP_NAME}/

exit

# # submission commands
# # partitions: GPU, GPU-shared, GPU-small
# H=${1}
# N=${2}
# interact -t ${H}:00:00 --gres=gpu:v100-32:${N} -N 1 -p GPU-shared # OR
# salloc -J Interact --gres=gpu:v100-32:$(N) --nodes=1 --time=$(H):00:00 --partition=GPU-shared
#