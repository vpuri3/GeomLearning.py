#
export CUDA_VISIBLE_DEVICES="1,2,3"

# # restart from checkpoint
# EXP_NAME="onecycle"

# torchrun \
#     --nproc-per-node gpu \
#     -m am \
#     --config out/am/${EXP_NAME}/config.yaml \
#     --restart_file out/am/${EXP_NAME}/ckpt02/model.pt

EXP_NAME="onecycle"

torchrun \
    --nproc-per-node gpu \
    -m am \
    --exp_name ${EXP_NAME} \
    --train true \
    --epochs 200 \
    --timeseries true \
    --sdf true \
    --TRA true \
    --tra_width 128 \
    --tra_num_heads 8 \
    --tra_num_slices 64 \
    --tra_num_layers 8 \
    --weight_decay 1e-4 \
    | tee -a out/am/${EXP_NAME}_log.txt 2>&1

# mv out/am/${EXP_NAME}_log.txt out/am/${EXP_NAME}/logfile.txt
#