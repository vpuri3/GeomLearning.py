#
export CUDA_VISIBLE_DEVICES="2"

# # restart from checkpoint
# EXP_NAME="ts"
# torchrun --nproc-per-node gpu -m am \
#     --config out/am/${EXP_NAME}/config.yaml \
#     --restart_file out/am/${EXP_NAME}/ckpt02/model.pt

EXP_NAME="exp"
torchrun --nproc-per-node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 3 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-3
