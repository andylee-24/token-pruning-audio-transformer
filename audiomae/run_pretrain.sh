#!/bin/bash

blr=2e-4 # Base Learning Rate (default value)


audioset_train_all_json=/mnt/dataset/audioset_2m_flac16k/as2m_16k.json
audioset_label=/root/AudioMAE-EE/dataset/audioset/class_labels_indices.csv

dataset=audioset
output_dir="exp-pretrain-small"
log_dir=${output_dir}/tensorboard-log

export OMP_NUM_THREADS=20

export TORCHELASTIC_ERROR_FILE=error.json

torchrun --nproc-per-node=2 main_pretrain.py \
    --blr $blr \
    --batch_size 128 \
    --num_workers 16 \
    --accum_iter 1 \
    --model mae_vit_small_patch16 \
    --mask_ratio 0.8 \
    --epochs 33 \
    --warmup_epochs 3 \
    --save_every_epoch 2 \
    --blr $blr \
    --weight_decay 0.0001 \
    --dataset $dataset \
    --data_train $audioset_train_all_json \
    --label_csv $audioset_label \
    --roll_mag_aug True \
    --decoder_mode 1 \
    --output_dir $output_dir \
    --log_dir $log_dir

# patch overlap X -> fshape = 128, tshape = 2
# --norm_pix_loss True \
