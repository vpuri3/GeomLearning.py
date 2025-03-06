#

EXP_NAME="ts0" # cts improvement ckpt10 -> 6.14-5
torchrun --master_port 29501 \
    --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 2 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-4 --learning_rate 1e-4

EXP_NAME="ts1" # ckptp10 -> 6.83e-5
torchrun --master_port 29501 \
    --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 2 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-3 --learning_rate 1e-4

EXP_NAME="ts2" # ckpt10 -> 7.89e-5
torchrun --master_port 29501 \
    --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 2 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-2 --learning_rate 1e-4

EXP_NAME="ts3" # ckpt08 -> 8.32e-5, ckpt10 -> 9.61e-5
torchrun --master_port 29501 \
    --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 2 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --weight_decay 1e-1 --learning_rate 1e-4

#---------------------------#

EXP_NAME="ts4" # ckpt10 -> 8.29e-5
torchrun --master_port 29501 \
    --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 2 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-4

EXP_NAME="ts5" # ckpt10 -> 2.00e-4
torchrun --master_port 29501 \
    --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 2 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-3

EXP_NAME="ts6" # ckpt10 -> 1.66e-4
torchrun --master_port 29501 \
    --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 2 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-2

EXP_NAME="ts7" # ckpt10 -> 7.00e-5
torchrun --master_port 29501 \
    --nproc_per_node gpu -m am --exp_name ${EXP_NAME} --train true \
    --epochs 200 --timeseries true --sdf true --TRA 2 --tra_width 128 \
    --tra_num_heads 8 --tra_num_slices 32 --tra_num_layers 8 \
    --schedule OneCycleLR --learning_rate 1e-3 --weight_decay 1e-1

#
