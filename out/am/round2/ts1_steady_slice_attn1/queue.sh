#

# EXP_NAME="ts0-steady" # stalled at 8.3e-4 ckpt08 - ckpt10 -> 8e-4
# torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
#     --epochs 200 --timeseries false --sdf true --TRA 1 --tra_width 128 \
#     --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
#     --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-3
# 
# #---------------------------#
# 
# EXP_NAME="ts1-steady" # ckpt10 -> 1.99e-3
# torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
#     --epochs 200 --timeseries false --sdf true --TRA 1 --tra_width 128 \
#     --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
#     --weight_decay 1e-3 --learning_rate 1e-4
# 
# EXP_NAME="ts2-steady" # 7.24e-4
# torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
#     --epochs 400 --timeseries false --sdf true --TRA 1 --tra_width 128 \
#     --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
#     --weight_decay 1e-3 --learning_rate 1e-4

#---------------------------#

EXP_NAME="ts0-steady"
torchrun --nproc_per_node gpu -m am \
    --config out/am/${EXP_NAME}/config.yaml \
    --restart_file out/am/${EXP_NAME}/ckpt10/model.pt \
    --schedule ConstantLR --learning_rate 1e-4 --weight_decay 1e-3 --epochs 400

EXP_NAME="ts1-steady"
torchrun --nproc_per_node gpu -m am \
    --config out/am/${EXP_NAME}/config.yaml \
    --restart_file out/am/${EXP_NAME}/ckpt10/model.pt \
    --schedule ConstantLR --learning_rate 1e-4 --weight_decay 1e-3 --epochs 400

#---------------------------#
#
