#!/bin/bash
set -euo pipefail

blr=1e-3
min_lr=1e-5
dataset="esc50"
nb_classes=50
epochs=120
shrink_start_epoch=20
shrink_epochs=40
first_eval_ep=60
batch_size=64
num_workers=4
warmup_epochs=4
mask_prob=0.3
if [ "$mask_prob" = "0.0" ]; then
    mask_2d=False
else
    mask_2d=True
fi
roll_mag_aug=True
mean_pooling=True
timem=96
freqm=24

# ./ramdisk.sh
readonly ramdisk_dir="/tmp/ramdisk/esc50"
mkdir -p ${ramdisk_dir}

base_keep_rate=$1
seeds=(12 34 56 78 90)

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



result_path=train_result.txt

readonly exp_dir=./exp-vit_b/${dataset}-kr${base_keep_rate}-maskprob${mask_prob}-bs${batch_size}-ep${epochs}-sse${shrink_start_epoch}-sdr${shrink_epochs}-minlr${min_lr}

for((fold=1;fold<=5;fold++));
do
    echo 'now process fold'${fold}
    
    # set your own dataset and pretrained model path
    audioset_pretrained_model_path="/root/AudioMAE/pretrained_models/pretrained.pth"
    train_data=/root/ast/src/finetune/esc50/data/datafiles/esc_train_data_${fold}.json
    eval_data=/root/ast/src/finetune/esc50/data/datafiles/esc_eval_data_${fold}.json
    esc50_label="/root/ast/src/finetune/esc50/data/esc_class_labels_indices.csv"
    
    exp_dir_fold=${exp_dir}/fold-${fold}

    for seed in "${seeds[@]}"; do
        exp_dir_fold_seed=${exp_dir_fold}/seed-${seed}
        
        # set --nproc_per_node to 1 for evaluation / feature extraction / ablation study
        # set --nproc_per_node to 2 for training
        
        torchrun --nproc_per_node=2 main_finetune.py \
            --ramdisk_dir ${ramdisk_dir} \
            --output_dir ${exp_dir_fold_seed} \
            --model vit_base_patch16 \
            --dataset $dataset \
            --data_train $train_data \
            --data_eval $eval_data \
            --label_csv $esc50_label \
            --audioset_pretrained_model_path $audioset_pretrained_model_path \
            --nb_classes $nb_classes \
            --epochs ${epochs} \
            --blr $blr \
            --batch_size ${batch_size} \
            --num_workers ${num_workers} \
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
            --base_keep_rate ${base_keep_rate} \
            --shrink_epochs ${shrink_epochs} \
            --shrink_start_epoch ${shrink_start_epoch} \
            --first_eval_ep ${first_eval_ep} \
            --min_lr ${min_lr} \
            --seed ${seed} \
            --result_path ${exp_dir_fold_seed}/${result_path}

            # custom_rank "mean" or "std" - set different exp_path

            # Ablation study: discarding C1, C2 tokens or C4, C5 tokens
            # --eval \
            # --finetuned_model_path $exp_dir_fold_seed/best_model.pth \
            # --retain_max ${retain_max} \
            # --retain_min ${retain_min} \
            # --drop_token_blk_idx ${drop_token_blk_idx} \
            # --result_path ${exp_dir_fold_seed}/${result_pattern}

            # Extracting features for visualization
            # --eval \
            # --flag_extract_features True \
            # --finetuned_model_path $exp_dir_fold_seed/best_model.pth \
            # --extract_features_path $exp_dir_fold_seed/extracted_features

        if [ $? -ne 0 ]; then
            echo "Error occurred (torchrun)"
            exit 1
        fi
    done
done

python average_esc50_score.py ${exp_dir} --pattern ${result_path}