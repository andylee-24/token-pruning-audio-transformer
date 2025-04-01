#!/bin/bash
set -euo pipefail

readonly model=ast
readonly dataset=audioset
readonly set=balanced
readonly imagenetpretrain=True
readonly audiosetpretrain=True
if [ $set == balanced ]
then
  readonly bal=none
  readonly lr=1e-4
  readonly epoch=30
  readonly tr_data=/mnt/shared/alpaca/audioset_20k_flac16k/as20k_16k.json
  readonly lrscheduler_start=10
  readonly lrscheduler_step=5
  readonly lrscheduler_decay=0.5
else
  echo "FINETUNE ONLY"
  exit 1
fi
readonly te_data=/mnt/shared/alpaca/audioset_eval_flac16k/audioset_eval.json
readonly mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
readonly fstride=16
readonly tstride=16
readonly batch_size=64


readonly dataset_mean=-4.2677393
readonly dataset_std=4.5689974

readonly audio_length=1024
readonly noise=False

readonly metrics=mAP
readonly loss=BCE
readonly warmup=True

keep_rate=$1

# drop_token_blk_idx=$2
readonly ramdisk_dir="/tmp/ramdisk_dir/as20k"
mkdir -p $ramdisk_dir

readonly audioset_pretrained_model_path="/root/evit_ast/pretrained_models/audioset_16_16_0.4422.pth"
readonly exp_dir=./exp-tpma2-ast/${dataset}-kr${keep_rate}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}


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



CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir --ramdisk_dir $ramdisk_dir \
  --label-csv ./data/class_labels_indices.csv --n_class 527 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
  --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
  --audioset_pretrained_model_path ${audioset_pretrained_model_path} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
  --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
  --shrink_start_epoch 15 --shrink_epochs 10 --base_keep_rate ${keep_rate} \
  --first_eval_epoch 25 \
  --model_size "base384" \
  --num-workers 8

  # intensity / variance based token pruning
  # custom_rank "mean" or "std" - set different exp_path

  # for feature extraction, enable export CUDA_VISIBLE_DEVICES=0
  # --eval \
  # --flag_extract_features True \
  # --extract_features_path $exp_dir/extracted_features

  # Ablation study: discarding C1, C2 tokens or C4, C5 tokens (models trained with keep-rate=1.0)
  # enable export CUDA_VISIBLE_DEVICES=0
  # --eval \
  # --retain_max ${retain_max} \
  # --retain_min ${retain_min} \
  # --drop_token_blk_idx ${drop_token_blk_idx} \
  # --eval_result_path ${exp_dir}/${result_pattern}