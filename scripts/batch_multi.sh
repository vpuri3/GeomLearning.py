#!/bin/bash

#SBATCH -N 2
#SBATCH -t 8:00:00
#SBATCH -p GPU
#SBATCH --gpus=v100-32:16
#SBATCH --reservation=GPUeng170006p

set -x
source $HOME/.bash_profile
cd /ocean/projects/eng170006p/vpuri1/GeomLearning.py
module load cuda anaconda3
conda activate GeomLearning

# Get the master node (first node in SLURM_NODELIST)
MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
NNODES=${SLURM_NNODES}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE}  # Number of GPUs per node

# Set the port for rendezvous
MASTER_PORT=29500  # Ensure this port is open on the master node

EXP_NAME="tra_timeseries_sdf"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m am \
    --exp_name ${EXP_NAME} \
    --sdf true \
    --train true \
    --timeseries true \
    --TRA true \
    --epochs 500 \
    --weight_decay 1e-2

cp slurm-${SLURM_JOB_ID}.out out/am/${EXP_NAME}/

exit
#