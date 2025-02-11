#
export CUDA_VISIBLE_DEVICES="1,2,3"

# torchrun --nproc-per-node gpu -m am --exp_name tra_timeseries \
#     --eval true --timeseries true

torchrun --nproc-per-node gpu -m am --exp_name gnn_timeseries_baseline \
    --eval true --timeseries true

torchrun --nproc-per-node gpu -m am --exp_name tra_timeseries_baseline \
    --eval true --timeseries true