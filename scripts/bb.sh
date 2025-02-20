#
export CUDA_VISIBLE_DEVICES="1,2,3"

# restart from checkpoint
# torchrun --nproc-per-node gpu -m am --config out/am/{EXP_NAME}/config.yaml \ --epochs 100

# timeseries with SDF
torchrun --nproc-per-node gpu -m am --exp_name tra_timeseries_sdf \
    --sdf true \
    --train true --timeseries true --TRA true --epochs 500 --weight_decay 1e-2

# # steady state with SDF
# torchrun --nproc-per-node gpu -m am --exp_name tra_steady_sdf \
#     --sdf true \
#     --train true --timeseries false --TRA true --epochs 500 --weight_decay 1e-2

# # Transolver w/o interpolation
# torchrun --nproc-per-node gpu -m am --exp_name tra_timeseries_weight_decay_0p01 \
#     --mask true --blend false --interpolate false \
#     --train true --timeseries true --TRA true --weight_decay 1e-2 \
#     --epochs 200

# # Baseline Transolver
# torchrun --nproc-per-node gpu -m am --exp_name tra_timeseries_baseline \
#     --mask false --blend false --interpolate false \
#     --train true --timeseries true --TRA true --weight_decay 1e-2

# # Baseline MeshGNN
# torchrun --nproc-per-node gpu -m am --exp_name gnn_timeseries_baseline \
#     --mask false --blend false --interpolate false \
#     --train true --timeseries true --GNN true --weight_decay 1e-3

#