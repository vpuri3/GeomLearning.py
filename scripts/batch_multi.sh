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

MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
NNODES=${SLURM_NNODES}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE}
MASTER_PORT=29500

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Number of Nodes: $NNODES"
echo "GPUs per Node: $NPROC_PER_NODE"

TORCHDISTRIBUTED_DEBUG=DETAIL torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    minimal_test.py #&

# sleep 120
# cp slurm-${SLURM_JOB_ID}.out out/am/${EXP_NAME}/
# wait


exit
#