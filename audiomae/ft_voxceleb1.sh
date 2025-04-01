#!/bin/bash
set -euo pipefail
blr=1e-3
min_lr=1e-5

model=vit_base_patch16

dataset="voxceleb1"
nb_classes=1251
epochs=90
shrink_start_epoch=20
shrink_epochs=40
first_eval_ep=60
base_keep_rate=$1
batch_size=32
num_workers=8
warmup_epochs=4
mask_prob=0.0

# if mask_prob > 0.0 then mask_prob = True else False
if [ "$mask_prob" = "0.0" ]; then
    mask_2d=False
else
    mask_2d=True
fi

timem=192
freqm=48

roll_mag_aug=True
mean_pooling=True

# VoxCeleb-1 uses a separate dataloder, set your own dataset and pretrained model path
voxceleb1_root=/mnt/shared/alpaca/voxceleb1
audioset_pretrained_model_path="/root/AudioMAE/pretrained_models/pretrained.pth"
train_data=dummy
eval_data=dummy
dummy_label=/root/TPMA2/util/voxceleb1_label.csv

# K-Means clustering results for VoxCeleb-1
# --- After re-mapping ---
# Cluster 0 => min: -1.5570, max: -1.0840
# Cluster 1 => min: -1.0840, max: -0.3911
# Cluster 2 => min: -0.3911, max: 0.0314
# Cluster 3 => min: 0.0314, max: 0.4389
# Cluster 4 => min: 0.4389, max: 1.8078

# retain_min=-0.3911
# retain_max=999
# drop_token_blk_idx=$2
# result_pattern=drop-c1c2-${drop_token_blk_idx}_${retain_min}_${retain_max}.txt


exp_dir=./exp-vit_b/${dataset}-kr${base_keep_rate}-maskprob${mask_prob}-bs${batch_size}-ep${epochs}-sse${shrink_start_epoch}-sdr${shrink_epochs}-minlr${min_lr}

# save the models in ram file system whlie training (./ramdisk.sh)
ramdisk_dir="/tmp/ramdisk/voxceleb1"
mkdir -p ${ramdisk_dir}

# set --nproc_per_node to 1 for evaluation / feature extraction / ablation study
# set --nproc_per_node to 2 for training

torchrun --nproc_per_node=2 main_finetune.py \
    --ramdisk_dir ${ramdisk_dir} \
    --output_dir $exp_dir \
    --model ${model} \
    --dataset $dataset \
    --data_train $train_data \
    --data_eval $eval_data \
    --label_csv $dummy_label \
    --num_workers ${num_workers} \
    --audioset_pretrained_model_path $audioset_pretrained_model_path \
    --nb_classes $nb_classes \
    --epochs ${epochs} \
    --blr $blr \
    --batch_size ${batch_size} \
    --warmup_epochs ${warmup_epochs} \
    --dist_eval \
    --mask_2d ${mask_2d} \
    --mask_t_prob ${mask_prob} \
    --mask_f_prob ${mask_prob} \
    --timem ${timem} \
    --freqm ${freqm} \
    --roll_mag_aug ${roll_mag_aug} \
    --min_lr ${min_lr} \
    --mean_pooling ${mean_pooling} \
    --voxceleb1_root ${voxceleb1_root} \
    --base_keep_rate ${base_keep_rate} \
    --shrink_epochs ${shrink_epochs} \
    --shrink_start_epoch ${shrink_start_epoch} \
    --first_eval_ep ${first_eval_ep} \
    --min_lr ${min_lr}
  
    # mean intensity / std based token pruning
    # --custom_rank "std" or "std" - set different exp_path

    # Ablation study: discarding C1, C2 tokens or C4, C5 tokens
    # --eval \
    # --finetuned_model_path $exp_dir/best_model.pth \
    # --retain_max ${retain_max} \
    # --retain_min ${retain_min} \
    # --drop_token_blk_idx ${drop_token_blk_idx} \
    # --result_path ${exp_dir}/${result_pattern}

    # Extracting features for visualization
    # --eval \
    # --flag_extract_features True \
    # --finetuned_model_path $exp_dir/best_model.pth \
    # --extract_features_path $exp_dir/extracted_features