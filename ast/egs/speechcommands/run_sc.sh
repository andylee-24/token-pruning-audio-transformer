#!/bin/bash
set -euo pipefail


model=ast
dataset=speechcommands
imagenetpretrain=True
audiosetpretrain=True
bal=none
lr=2.5e-4
epoch=30
mixup=0.6
batch_size=128

dataset_mean=-6.845978
dataset_std=5.5654526
audio_length=128
noise=True

metrics=acc
loss=BCE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

tr_data=/root/ast/src/finetune/speechcommands_v2_35/data/datafiles/speechcommand_train_data.json
val_data=/root/ast/src/finetune/speechcommands_v2_35/data/datafiles/speechcommand_valid_data.json
eval_data=/root/ast/src/finetune/speechcommands_v2_35/data/datafiles/speechcommand_eval_data.json

keep_rate=$1


audioset_pretrained_model_path="/root/evit_ast/pretrained_models/audioset_16_16_0.4422.pth"


exp_dir=./exp-tpma2-ast/${dataset}-kr${keep_rate}-b$batch_size-lr${lr}
# use ramdisk.sh

ramdisk_dir="/tmp/ramdisk/spc-2/$exp_dir"
mkdir -p $ramdisk_dir


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

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir --ramdisk_dir $ramdisk_dir \
  --label-csv ./data/speechcommands_class_labels_indices.csv --n_class 35 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
  --mixup ${mixup} --bal ${bal} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
  --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
  --tstride 16 --fstride 16 --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
  --shrink_start_epoch 5 --shrink_epochs 15 --base_keep_rate ${keep_rate} \
  --first_eval_epoch 20 \
  --audioset_pretrained_model_path ${audioset_pretrained_model_path} \
  --model_size "base384" \
  --num-workers 16

  # custom_rank "mean" or "std" - set different exp_path

  # for feature extraction, enable export CUDA_VISIBLE_DEVICES=0
  # --eval \
  # --flag_extract_features True \
  # --extract_features_path $exp_dir/extracted_features/data

  # Ablation study: discarding C1, C2 tokens or C4, C5 tokens (models trained with keep-rate=1.0)
  # enable export CUDA_VISIBLE_DEVICES=0
  # --eval \
  # --retain_max ${retain_max} \
  # --retain_min ${retain_min} \
  # --drop_token_blk_idx ${drop_token_blk_idx} \
  # --eval_result_path ${exp_dir}/${result_pattern}