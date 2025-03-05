#

EXP_NAME="ts0" # conv at ckpt10 -> 7.64e-5
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 3 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-4 --learning_rate 1e-4

EXP_NAME="ts1" # conv at ckpt10 -> 7.15e-5
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 3 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-3 --learning_rate 1e-4

EXP_NAME="ts2" # conv at ckpt10 -> 7.59e-5
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 3 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-2 --learning_rate 1e-4

EXP_NAME="ts3" # ctd improvement but no conv so far. at ckpt10 -> 8.78e-5
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 3 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-1 --learning_rate 1e-4

#---------------------------#

EXP_NAME="ts4" # ctd improvement but no conv so far. at ckpt10 -> 1.98e-4
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 3 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-4

EXP_NAME="ts5" # conv at ckpt06 -> 6.27e-5 and cts improvement to ckpt10 -> 4.99e-5
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 3 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-3

EXP_NAME="ts6" # cts improvement. ckpt10 -> 7.27e-5
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 3 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-2

EXP_NAME="ts7"
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 3 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-1

#
