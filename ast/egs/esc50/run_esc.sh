#!/bin/bash
set -euo pipefail
model=ast
dataset=esc50
imagenetpretrain=True
audiosetpretrain=True
bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
mixup=0
epoch=30
batch_size=48

dataset_mean=-6.6268077
dataset_std=5.358466

audio_length=512
noise=False

metrics=acc
loss=CE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

keep_rate=$1

# drop_token_blk_idx=$2
# For esc-50, we set seed as 12, 34, 56, 78 and 90
# Define list of seeds

readonly seeds=(12 34 56 78 90)

# seeds=(12) # - We used this seed for feature extraction

# ESC-50 intensity-cluster info for ablation study
# Cluster 0 => min: -0.8692, max: -0.5829, size: 0.186
# Cluster 1 => min: -0.5829, max: -0.2063, size: 0.159
# Cluster 2 => min: -0.2063, max: 0.1130, size: 0.223
# Cluster 3 => min: 0.1130, max: 0.4303, size: 0.254
# Cluster 4 => min: 0.4303, max: 1.2028, size: 0.178

# retain_min=-999
# retain_max=0.1130
# drop_token_blk_idx=$2
# result_pattern=drop-c4c5-${drop_token_blk_idx}_${retain_min}_${retain_max}.txt

audioset_pretrained_model_path="/root/evit_ast/pretrained_models/audioset_16_16_0.4422.pth"

base_exp_dir=./exp-tpma2-ast/${dataset}-kr${keep_rate}-b$batch_size-lr${lr}


mkdir -p $base_exp_dir

for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir_fold=${base_exp_dir}/fold-${fold}

  tr_data=/root/ast/src/finetune/esc50/data/datafiles/esc_train_data_${fold}.json
  te_data=/root/ast/src/finetune/esc50/data/datafiles/esc_eval_data_${fold}.json

  for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed} for fold ${fold}"
    exp_dir_fold_seed=${exp_dir_fold}/seed-${seed}
    ramdisk_dir="/tmp/ramdisk" # use ramdisk.sh
    ramdisk_dir=${ramdisk_dir}/$exp_dir_fold_seed
    mkdir -p $ramdisk_dir
    CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
      --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir_fold_seed --ramdisk_dir $ramdisk_dir \
      --label-csv /root/ast/src/finetune/esc50/data/esc_class_labels_indices.csv --n_class 50 \
      --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
      --mixup ${mixup} --bal ${bal} \
      --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
      --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
      --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
      --shrink_start_epoch 5 --shrink_epochs 15 --base_keep_rate ${keep_rate} \
      --first_eval_epoch 20 \
      --audioset_pretrained_model_path ${audioset_pretrained_model_path} \
      --model_size "base384" \
      --num-workers 16 \
      --seed ${seed} \

      # custom_rank "mean" or "std" - set different exp_path

      # for feature extraction, enable export CUDA_VISIBLE_DEVICES=0
      # --eval \
      # --flag_extract_features True \
      # --extract_features_path $exp_dir_fold_seed/extracted_features
    
      # Ablation study: discarding C1, C2 tokens or C4, C5 tokens (models trained with keep-rate=1.0)
      # enable export CUDA_VISIBLE_DEVICES=0
      # --eval \
      # --retain_max ${retain_max} \
      # --retain_min ${retain_min} \
      # --drop_token_blk_idx ${drop_token_blk_idx} \
      # --eval_result_path ${exp_dir_fold_seed}/${result_pattern}

    # if exit not clean then halt
    if [ $? -ne 0 ]; then
      echo 'exit code is not 0'
      exit 1
    fi
  done
done

# python ./get_esc_result.py --exp_path ${base_exp_dir}
python ./get_esc_result.py --exp_path ${base_exp_dir} --result_file best_result.csv