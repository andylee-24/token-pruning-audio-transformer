#!/bin/bash
set -euo pipefail

blr=1e-3
min_lr=1e-5

dataset="spc2"
nb_classes=35
epochs=90
shrink_start_epoch=10
shrink_epochs=30
first_eval_ep=40
base_keep_rate=$1
batch_size=256
num_workers=8
warmup_epochs=4
mask_prob=0.0
if [ "$mask_prob" = "0.0" ]; then
    mask_2d=False
else
    mask_2d=True
fi
timem=48
freqm=48
mixup=0.5
roll_mag_aug=True
mean_pooling=True

# set your own dataset and pretrained model path
train_data="/root/ast/src/finetune/speechcommands_v2_35/data/datafiles/speechcommand_train_data.json"
eval_data="/root/ast/src/finetune/speechcommands_v2_35/data/datafiles/speechcommand_eval_data.json"
ks2_label="/root/ast/src/finetune/speechcommands_v2_35/data/speechcommands_class_labels_indices.csv"
audioset_pretrained_model_path="/root/AudioMAE/pretrained_models/pretrained.pth"


# SPC-2 intensity-cluster info for ablation study
# Cluster 0 => min: -0.8172, max: -0.6170
# Cluster 1 => min: -0.6170, max: -0.3210
# Cluster 2 => min: -0.3210, max: -0.0423
# Cluster 3 => min: -0.0423, max: 0.2596
# Cluster 4 => min: 0.2596, max: 1.0952

# retain_min=-999
# retain_max=-0.0423
# drop_token_blk_idx=$2
# result_pattern=drop-c4c5-${drop_token_blk_idx}_${retain_min}_${retain_max}.txt


exp_dir=./exp-vit_b/${dataset}-kr${base_keep_rate}-maskprob${mask_prob}-bs${batch_size}-ep${epochs}-sse${shrink_start_epoch}-sdr${shrink_epochs}-minlr${min_lr}
finetuned_model_path="$exp_dir/best_model.pth"

# save the models in ram file system while training (./ramdisk.sh)
ramdisk_dir="/tmp/ramdisk/spc-2"
mkdir -p $ramdisk_dir

# set --nproc_per_node to 1 for evaluation / feature extraction / ablation study

torchrun --nproc_per_node=2 main_finetune.py \
    --ramdisk_dir ${ramdisk_dir} \
    --output_dir $exp_dir \
    --model vit_base_patch16 \
    --dataset $dataset \
    --data_train $train_data \
    --data_eval $eval_data \
    --label_csv $ks2_label \
    --audioset_pretrained_model_path $audioset_pretrained_model_path \
    --nb_classes $nb_classes \
    --epochs ${epochs} \
    --blr $blr \
    --batch_size ${batch_size} \
    --warmup_epochs ${warmup_epochs} \
    --num_workers ${num_workers} \
    --dist_eval \
    --mask_2d ${mask_2d} \
    --mask_t_prob ${mask_prob} \
    --mask_f_prob ${mask_prob} \
    --timem ${timem} \
    --freqm ${freqm} \
    --roll_mag_aug ${roll_mag_aug} \
    --min_lr ${min_lr} \
    --mixup ${mixup} \
    --mean_pooling ${mean_pooling} \
    --base_keep_rate ${base_keep_rate} \
    --shrink_epochs ${shrink_epochs} \
    --shrink_start_epoch ${shrink_start_epoch} \
    --first_eval_ep ${first_eval_ep}

    # custom_rank "mean" or "std" - set different exp_path

    # Extracting features for visualization
    # --eval \
    # --flag_extract_features True \
    # --finetuned_model_path $exp_dir/best_model.pth \
    # --extract_features_path $exp_dir/extracted_features

    # Ablation study: discarding C1, C2 tokens or C4, C5 tokens
    # --eval \
    # --finetuned_model_path ${fine_tuned_model_path} \
    # --retain_max ${retain_max} \
    # --retain_min ${retain_min} \
    # --drop_token_blk_idx ${drop_token_blk_idx} \
    # --result_path ${exp_dir}/${result_pattern}
