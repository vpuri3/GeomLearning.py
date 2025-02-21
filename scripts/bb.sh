#
export CUDA_VISIBLE_DEVICES="1,2,3"

# # restart from checkpoint
# EXP_NAME=""
# torchrun \
#     --nproc-per-node gpu \
#     -m am \
#     --config out/am/{EXP_NAME}/config.yaml \
#     --epochs 100

EXP_NAME="tra_timeseries_sdf_layers_5_width_128_slices_64_wd_0p01_heads_8"

torchrun \
    --nproc-per-node gpu \
    -m am \
    --exp_name ${EXP_NAME} \
    --train true \
    --epochs 500 \
    --timeseries true \
    --sdf true \
    --TRA true \
    --tra_width 128 \
    --tra_num_heads 8 \
    --tra_num_slices 64 \
    --tra_num_layers 5 \
    --weight_decay 1e-2

#