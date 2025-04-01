#!/bin/bash
feature_path=$1

python extract_stats.py \
    --feature_dict_path $feature_path \
    --output_dir $feature_path \
    --retained_token_analyze
    # --plt_title "VoxCeleb-1 / keep-rate 0.5"
    # relative_attention_score