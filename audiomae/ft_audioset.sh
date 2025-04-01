#!/bin/bash
set -euo pipefail

blr=1e-3
min_lr=1e-5

# set your own dataset and pretrained model path
audioset_pretrained_model_path="/root/AudioMAE/pretrained_models/pretrained.pth"
audioset_train_json="/mnt/dataset/audioset_20k_flac16k/as20k_16k.json"
audioset_eval_json="/mnt/dataset/audioset_eval_flac16k/audioset_eval.json"
audioset_label="/root/AudioMAE/dataset/audioset/class_labels_indices.csv"

dataset=audioset
nb_classes=527
roll_mag_aug=True
mean_pooling=True

mask_prob=0.3
mask_2d=True
timem=192
freqm=48
batch_size=16
epochs=60
shrink_start_epoch=30
shrink_epochs=20
first_eval_ep=50
base_keep_rate=$1
mixup=0.5

# ./ramdisk.sh
ramdisk_dir="/tmp/ramdisk/as20k"
mkdir -p $ramdisk_dir

# AS-20K intensity-cluster info for ablation study
# Cluster 0 => min: -1.2776, max: -0.8705, size: 0.094
# Cluster 1 => min: -0.8705, max: -0.3861, size: 0.162
# Cluster 2 => min: -0.3861, max: -0.0182, size: 0.262
# Cluster 3 => min: -0.0182, max: 0.3140, size: 0.295
# Cluster 4 => min: 0.3140, max: 1.2121, size: 0.187

# retain_min=-999
# retain_max=-0.0182
# drop_token_blk_idx=$2
# result_pattern=drop-c4c5-${drop_token_blk_idx}_${retain_min}_${retain_max}.txt


exp_dir=./exp-vit_b/${dataset}-kr${base_keep_rate}-maskprob${mask_prob}-bs${batch_size}-ep${epochs}-sse${shrink_start_epoch}-sdr${shrink_epochs}-minlr${min_lr}
finetuned_model_path="$exp_dir/best_model.pth"

torchrun --nproc_per_node=2 main_finetune.py \
    --ramdisk_dir ${ramdisk_dir} \
    --output_dir ${exp_dir} \
    --model vit_base_patch16 \
    --dataset $dataset \
    --data_train $audioset_train_json \
    --data_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --audioset_pretrained_model_path $audioset_pretrained_model_path \
    --roll_mag_aug ${roll_mag_aug} \
    --epochs ${epochs} \
    --blr $blr \
    --batch_size ${batch_size} \
    --warmup_epochs 4 \
    --dist_eval \
    --num_workers 4 \
    --mask_2d ${mask_2d} \
    --mask_t_prob ${mask_prob} \
    --mask_f_prob ${mask_prob} \
    --timem ${timem} \
    --freqm ${freqm} \
    --mixup ${mixup} \
    --mean_pooling ${mean_pooling} \
    --nb_classes ${nb_classes} \
    --first_eval_ep ${first_eval_ep} \
    --base_keep_rate ${base_keep_rate} \
    --shrink_epochs ${shrink_epochs} \
    --shrink_start_epoch ${shrink_start_epoch} \
    --min_lr ${min_lr}

    # custom_rank "mean" or "std" - set different exp_path

    # Extracting features for visualization
    # --eval \
    # --flag_extract_features True \
    # --finetuned_model_path $exp_dir/best_model.pth \
    # --extract_features_path $exp_dir/extracted_features \

    # Ablation study: discarding C1, C2 tokens or C4, C5 tokens
    # --eval \
    # --finetuned_model_path $exp_dir/best_model.pth \
    # --retain_max ${retain_max} \
    # --retain_min ${retain_min} \
    # --drop_token_blk_idx ${drop_token_blk_idx} \
    # --result_path ${exp_dir}/${result_pattern}