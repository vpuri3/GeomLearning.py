#

EXP_NAME="ts0"
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA true --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-4 --learning_rate 1e-4

EXP_NAME="ts1"
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA true --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-2 --learning_rate 1e-4

EXP_NAME="ts2"
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA true --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-1 --learning_rate 1e-4

EXP_NAME="ts3"
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA true --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-2 --learning_rate 5e-4

#---------------------------#

EXP_NAME="ts4"
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA true --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 0e-3

EXP_NAME="ts5"
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA true --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-3

EXP_NAME="ts6"
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA true --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-2

EXP_NAME="ts7"
torchrun --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA true --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-1

#
