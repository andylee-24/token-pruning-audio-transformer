import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from matplotlib.pyplot import get_cmap
from matplotlib import colors

from util.aggregate_features import combine_feature_dict
from util.token_reduction_utils import get_melspec_idx, plot_token_occurrence_heatmap
from util.misc import save_melspec_batch, apply_mask
from typing import Union 
from einops import rearrange
from tqdm import tqdm

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
    

def visualize_mask(feature_dict_path: Union[str, Path], output_dir: Union[str, Path]):
    
    visualize_output_path = os.path.join(output_dir, 'visualize')
    os.makedirs(visualize_output_path, exist_ok=True)
    
    melspec_batch = combine_feature_dict(feature_dict_path, 'mel')    
    labels = combine_feature_dict(feature_dict_path, 'labels')
    
    block_3_topk_idx = combine_feature_dict(feature_dict_path, 'block-3.topk_idx')
    block_6_topk_idx = combine_feature_dict(feature_dict_path, 'block-6.topk_idx')
    block_9_topk_idx = combine_feature_dict(feature_dict_path, 'block-9.topk_idx')
    
    num_items = 64
    melspec_batch = melspec_batch[:num_items]
    labels = labels[:num_items]
    block_3_topk_idx = block_3_topk_idx[:num_items]
    block_6_topk_idx = block_6_topk_idx[:num_items]
    block_9_topk_idx = block_9_topk_idx[:num_items]
    
    B, _, T, F = melspec_batch.shape
    # assert T >= F, "Time dimension must be greater than frequency dimension"
    melspec_batch = melspec_batch.squeeze(1)  # (B, 1, T, F) to (B, T, F)
    cmap = get_cmap('viridis')
    melspec_batch = cmap(melspec_batch)  # The shape will be (B, H, W, 4) including alpha channel (dtype: float32)
    melspec_batch = torch.from_numpy(melspec_batch[..., :3]) # (B, H, W, 4) to (B, H, W, 3)
    melspec_batch = melspec_batch.permute(0, 3, 1, 2)  # (B, H, W, 3) to (B, 3, H, W)
    
    assert (block_3_topk_idx is not None) and (block_6_topk_idx is not None) and (block_9_topk_idx is not None)
    
    topk_reduced_idx_list = [block_3_topk_idx, block_6_topk_idx, block_9_topk_idx]
    
    topk_melspec_idx_list = get_melspec_idx(topk_reduced_idx_list)

    save_melspec_batch(melspec_batch, visualize_output_path, 
                       file_name='melspec_{}_{}_input.jpg',
                       start_idx=0, labels=labels)
        
    for pruning_stage_idx, topk_melspec_idx in enumerate(topk_melspec_idx_list):
        masked_melspec = apply_mask(melspec_batch, patch_size=16, idx=topk_melspec_idx)
        save_melspec_batch(masked_melspec, visualize_output_path, 
                           file_name='melspec_{}_{}' + f'_{pruning_stage_idx}.jpg', 
                           start_idx=0, labels=labels)
 

    

def kendall_rank(feature_dict_path: Union[str, Path], output_dir: Union[str, Path], stat: str, fig_title: str):
        
    # feature_dict = load_feature_dicts(feature_dict_path, allowlist=allowlist, nproc=4, load_subset=False)
    mel_specs = combine_feature_dict(feature_dict_path, 'mel', load_subset=False)
    # B, _, T, F = melspec.shape

    h = mel_specs.shape[2] // 16
    w = mel_specs.shape[3] // 16
    
    if stat == 'mean':
        melspec_patch_stat = rearrange(mel_specs, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h, w=w).mean(dim=1)
        n_clusters = 5

    elif stat == 'std':
        melspec_patch_stat = rearrange(mel_specs, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h, w=w).std(dim=1)
        n_clusters = 5

    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    melspec_rank = kmeans.fit_predict(melspec_patch_stat.reshape(-1, 1)).reshape(melspec_patch_stat.shape)
    melspec_rank = torch.from_numpy(melspec_rank).float()

    # 1) Compute stats for each cluster, ** cluster is unordered with respect to the statistics (mean intesnity)
    cluster_info_list = []
    csize_sum = 0
    for cluster_id in range(n_clusters):
        cluster_mask = (melspec_rank == cluster_id)
        cluster_values = melspec_patch_stat[cluster_mask]
        cmin = cluster_values.min().item()
        cmax = cluster_values.max().item()
        csize = cluster_values.size()
        cluster_info_list.append((cluster_id, cmin, cmax, csize))
        csize_sum += csize[0]

    print("\n--- Original cluster stats ---")
    for (cluster_id, cmin, cmax, csize) in cluster_info_list:
        print(f"Cluster {cluster_id} => min: {cmin:.4f}, max: {cmax:.4f}")

    # 2) Sort them by cmin ascending
    sorted_cluster_info_list = sorted(cluster_info_list, key=lambda x: x[1])  

    # 3) Figure out the global min/max across *all* clusters
    all_mins = [stat[1] for stat in sorted_cluster_info_list]
    all_maxs = [stat[2] for stat in sorted_cluster_info_list]
    global_min = min(all_mins)
    global_max = max(all_maxs)

    # 2) Create figure + axis
    fig, ax = plt.subplots(figsize=(8, 0.8))

    # 3) Build a simple horizontal gradient from 0 to 1
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))  # shape: (2, 256)

    # 4) "Draw" the gradient as an image from global_min to global_max
    #    so it looks like a color bar
    img = ax.imshow(
        gradient,
        extent=[global_min, global_max, 0, 1],  # x from gmin->gmax, y from 0->1
        aspect='auto', 
        cmap='viridis'
    )

    # Hide the y-axis, since it's just 0..1
    ax.set_yticks([])
    ax.set_ylim([0, 1])

    # Label the x-axis
    ax.set_xlabel(f'{fig_title}', fontsize=16)

    # 5) Draw vertical lines for each cluster’s min and max
    for idx, (cluster_id, cmin, cmax, csize) in enumerate(sorted_cluster_info_list):
        # Draw dashed lines for cluster boundaries
        ax.axvline(x=cmin, color='black', linestyle='-', linewidth=2)
        ax.axvline(x=cmax, color='black', linestyle='-', linewidth=2)
        
        midpoint = 0.5 * (cmin + cmax)
        ax.text(
            midpoint, 1.02, f"C{idx+1}: {int((csize[0]/csize_sum)*100)}%",
            ha='center', va='bottom', color='black', fontsize=14,
            transform=ax.get_xaxis_transform()  # positions in data coords along X, 0..1 in Y
        )

    # 6) Adjust and show
    ax.set_xlim(global_min, global_max)
    plt.tight_layout(pad=0)
    
    if stat == 'mean':
        plt.savefig(f'kmean-{fig_title}.jpg')
        plt.savefig(f'kmean-{fig_title}.eps', format='eps', pad_inches=0, bbox_inches='tight')
    
    # 3) Build mapping old_id -> new_id
    old_to_new = {}
    for new_cluster_id, (old_cluster_id, cmin, cmax, csize) in enumerate(sorted_cluster_info_list):
        old_to_new[old_cluster_id] = new_cluster_id

    # 4) Apply new mapping to get "sorted" cluster IDs
    new_melspec_rank = torch.zeros_like(melspec_rank)
    for old_cluster_id, new_cluster_id in old_to_new.items():
        new_melspec_rank[melspec_rank == old_cluster_id] = new_cluster_id

    melspec_rank = new_melspec_rank  # If you want to overwrite it

    
    # Assign clusters based on your manual thresholds.
    # color print (Warning for manual assignment using termcolor
    # from termcolor import colored
    # print(colored("WARNING - Manual assignment of clusters based on thresholds", "red"))
    # assert stat == "std" and "audioset" in feature_dict_path, "Manual assignment is only for audioset"
    # # import pdb; pdb.set_trace()
    # melspec_rank[(melspec_patch_stat >= 0.0)   & (melspec_patch_stat < 0.1151)] = 0
    # melspec_rank[(melspec_patch_stat >= 0.1151) & (melspec_patch_stat < 0.2011)] = 1
    # melspec_rank[(melspec_patch_stat >= 0.2011) & (melspec_patch_stat < 0.3223)] = 2
    # melspec_rank[(melspec_patch_stat >= 0.3223) & (melspec_patch_stat < 0.5401)] = 3
    # melspec_rank[(melspec_patch_stat >= 0.5401) & (melspec_patch_stat <= 1.1858)] = 4

    # 5) (Optional) Recompute stats in the new ordering
    new_cluster_info_list = []
    for cluster_id in range(n_clusters):
        cluster_mask = (melspec_rank == cluster_id)
        cluster_values = melspec_patch_stat[cluster_mask]
        cmin = cluster_values.min().item()
        cmax = cluster_values.max().item()
        cluster_size = cluster_values.size()
        new_cluster_info_list.append((cluster_id, cmin, cmax, cluster_size))

    print("\n--- After re-mapping ---")
    for (cluster_id, cmin, cmax, cluster_size) in new_cluster_info_list:
        print(f"Cluster {cluster_id} => min: {cmin:.4f}, max: {cmax:.4f}, size: {cluster_size[0]/csize_sum:.3f}")
    # assert False
    kendall_rank_list = []
    for blk_id in tqdm(range(0, 12)):
            
        attn_score_batch = combine_feature_dict(feature_dict_path, f'block-{blk_id}.attn_score')
        
        batch_size, num_tokens = attn_score_batch.shape
        
        num_concordant_pairs = 0
        num_discordant_pairs = 0
        
        
        for idx_batch in tqdm(range(batch_size)):
            # Extract scores and ranks for the current batch
            attn_score = attn_score_batch[idx_batch]
            rank = melspec_rank[idx_batch]
            
            # Compute pairwise differences
            attn_diff = attn_score.unsqueeze(0) - attn_score.unsqueeze(1)  # Shape: [num_tokens, num_tokens]
            rank_diff = rank.unsqueeze(0) - rank.unsqueeze(1)              # Shape: [num_tokens, num_tokens]
            
            # Create upper triangular mask to avoid redundant comparisons and self-comparisons
            mask = torch.triu(torch.ones_like(attn_diff, dtype=torch.bool), diagonal=1)  # Shape: [num_tokens, num_tokens]
            
            # Concordant and discordant conditions
            concordant = ((attn_diff * rank_diff) >= 0) & mask
            discordant = ((attn_diff * rank_diff) < 0) & mask
            
            # Count pairs
            num_concordant_pairs += concordant.sum().item()
            num_discordant_pairs += discordant.sum().item()

        num_pairs = ((num_tokens * (num_tokens - 1)) // 2) * batch_size
        kendall_coefficient = (num_concordant_pairs - num_discordant_pairs) / num_pairs
        print(f'\nKendall Rank for block {blk_id}: {kendall_coefficient}')
        assert (kendall_coefficient >= -1 and kendall_coefficient <= 1)
        kendall_rank_list.append(kendall_coefficient)

        # if token pruning occurs, remove corresponding tokens in the mel spectrogram
        topk_reduced_idx = combine_feature_dict(feature_dict_path, f'block-{blk_id}.topk_idx')
        if topk_reduced_idx is not None:
            melspec_rank = torch.gather(melspec_rank, dim=1, index=topk_reduced_idx)
    
    print(feature_dict_path)
    print(stat)
    print(kendall_rank_list)
        


          
            
    

def extract_kmeans_rank(melspec_patch_stat: torch.Tensor, n_clusters: int = 5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    melspec_rank = kmeans.fit_predict(melspec_patch_stat.reshape(-1, 1)).reshape(melspec_patch_stat.shape)
    melspec_rank = torch.from_numpy(melspec_rank).float()

    # 1) Compute stats for each cluster, ** cluster is unordered with respect to the statistics (mean intesnity)
    cluster_info_list = []
    csize_sum = 0
    for cluster_id in range(n_clusters):
        cluster_mask = (melspec_rank == cluster_id)
        cluster_values = melspec_patch_stat[cluster_mask]
        cmin = cluster_values.min().item()
        cmax = cluster_values.max().item()
        csize = cluster_values.size()
        cluster_info_list.append((cluster_id, cmin, cmax, csize))
        csize_sum += csize[0]

    # 2) Sort them by cmin ascending
    sorted_cluster_info_list = sorted(cluster_info_list, key=lambda x: x[1])  

    # 3) Build mapping old_id -> new_id
    old_to_new = {}
    for new_cluster_id, (old_cluster_id, cmin, cmax, csize) in enumerate(sorted_cluster_info_list):
        old_to_new[old_cluster_id] = new_cluster_id

    # 4) Apply new mapping to get "sorted" cluster IDs
    new_melspec_rank = torch.zeros_like(melspec_rank)
    for old_cluster_id, new_cluster_id in old_to_new.items():
        new_melspec_rank[melspec_rank == old_cluster_id] = new_cluster_id

    return new_melspec_rank


    
def retained_token_visualize(feature_dict_path: Union[str, Path], output_dir: Union[str, Path]):
    if 'esc50' in feature_dict_path:
        fig_title = 'ESC-50'
    elif 'audioset' in feature_dict_path:
        fig_title = 'AS-20K'
    elif 'spc' in feature_dict_path or 'speech' in feature_dict_path:
        fig_title = 'SPC-2'
    elif 'voxceleb' in feature_dict_path:
        fig_title = 'VoxCeleb-1'
    else:
        assert False, "Invalid dataset"
    
    if 'kr1.0' in feature_dict_path:
        fig_title += '_input'
    elif 'kr0.5' in feature_dict_path:
        fig_title += '_kr0.5'

    # feature_dict = load_feature_dicts(feature_dict_path, allowlist=allowlist, nproc=4, load_subset=False)
    mel_specs = combine_feature_dict(feature_dict_path, 'mel', load_subset=False)
    # B, _, T, F = melspec.shape

    h = mel_specs.shape[2] // 16
    w = mel_specs.shape[3] // 16
    
    melspec_patch_mean = rearrange(mel_specs, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h, w=w).mean(dim=1)
    melspec_patch_std = rearrange(mel_specs, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h, w=w).std(dim=1)

    n_cluster = 5
    melspec_rank_mean = extract_kmeans_rank(melspec_patch_mean, n_clusters=n_cluster)
    melspec_rank_std = extract_kmeans_rank(melspec_patch_std, n_clusters=n_cluster)

    for blk_id in tqdm(range(0, 12)):
            
        # attn_score_batch = combine_feature_dict(feature_dict_path, f'block-{blk_id}.attn_score')

        # if token pruning occurs, remove corresponding tokens in the mel spectrogram
        topk_reduced_idx = combine_feature_dict(feature_dict_path, f'block-{blk_id}.topk_idx')
        if topk_reduced_idx is not None:
            melspec_rank_mean = torch.gather(melspec_rank_mean, dim=1, index=topk_reduced_idx)
            melspec_rank_std = torch.gather(melspec_rank_std, dim=1, index=topk_reduced_idx)
            melspec_patch_mean = torch.gather(melspec_patch_mean, dim=1, index=topk_reduced_idx)
            melspec_patch_std = torch.gather(melspec_patch_std, dim=1, index=topk_reduced_idx)
    

    fig = plot_token_occurrence_heatmap(melspec_patch_mean.flatten(), melspec_patch_std.flatten(), bins=20, fig_title=f'{fig_title}' )
    
    fig.savefig(f'retain_token_stat_{fig_title}.jpg', dpi=600)
    fig.savefig(f'retain_token_stat_{fig_title}.eps', format='eps')
    


    
def retained_token_analyze(feature_dict_path: Union[str, Path], output_dir: Union[str, Path]):
    if 'esc50' in feature_dict_path:
        fig_title = 'ESC-50'
    elif 'audioset' in feature_dict_path:
        fig_title = 'AS-20K'
    elif 'spc' in feature_dict_path or 'speech' in feature_dict_path:
        fig_title = 'SPC-2'
    elif 'voxceleb' in feature_dict_path:
        fig_title = 'VoxCeleb-1'
    else:
        assert False, "Invalid dataset"
    
    if 'kr1.0' in feature_dict_path:
        fig_title += '_input'
    elif 'kr0.5' in feature_dict_path:
        fig_title += '_kr0.5'

    mel_specs = combine_feature_dict(feature_dict_path, 'mel', load_subset=False)
    # B, _, T, F = melspec.shape

    h = mel_specs.shape[2] // 16
    w = mel_specs.shape[3] // 16
    
    melspec_patch_mean = rearrange(mel_specs, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h, w=w).mean(dim=1)
    melspec_patch_std = rearrange(mel_specs, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h, w=w).std(dim=1)

    n_cluster = 5
    melspec_rank_mean = extract_kmeans_rank(melspec_patch_mean, n_clusters=n_cluster)
    melspec_rank_std = extract_kmeans_rank(melspec_patch_std, n_clusters=n_cluster)

    for blk_id in tqdm(range(0, 12)):
            
        # attn_score_batch = combine_feature_dict(feature_dict_path, f'block-{blk_id}.attn_score')

        # if token pruning occurs, remove corresponding tokens in the mel spectrogram
        topk_reduced_idx = combine_feature_dict(feature_dict_path, f'block-{blk_id}.topk_idx')
        if topk_reduced_idx is not None:
            melspec_rank_mean = torch.gather(melspec_rank_mean, dim=1, index=topk_reduced_idx)
            melspec_rank_std = torch.gather(melspec_rank_std, dim=1, index=topk_reduced_idx)
            melspec_patch_mean = torch.gather(melspec_patch_mean, dim=1, index=topk_reduced_idx)
            melspec_patch_std = torch.gather(melspec_patch_std, dim=1, index=topk_reduced_idx)

    # count the number of tokens both staisfying melspec_rank_mean < 2 and melspec_rank_std < 2
    retained_token = (melspec_rank_mean < 2)
    retained_token = retained_token.sum()
    print(retained_token)


def get_args():
    parser = argparse.ArgumentParser(description='Extract stats from the output file')
    # model feature directory
    parser.add_argument('--feature_dict_path', type=str, help='Path to the feature')
    # output directory
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    # visualization
    parser.add_argument('--visualize_mask', action='store_true', help='Visualize mel spectrogram')
    # intensity histogram - over audios
    parser.add_argument('--fig_title', type=str, help='Title of the histogram')
    # kendall-rank
    parser.add_argument('--kendall_rank_mean', action='store_true', help='Calculate kendall rank')
    parser.add_argument('--kendall_rank_std', action='store_true', help='Calculate kendall rank')
    parser.add_argument('--retained_token_visualize', action='store_true', help='Calculate retained token analysis')
    parser.add_argument('--retained_token_analyze', action='store_true', help='Calculate retained token analysis')
    parser.add_argument('--plt_title', type=str, help='Title of the plot', default='Attention score ratio')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    # collect feature dict
    if args.visualize_mask:
        visualize_mask(args.feature_dict_path, args.output_dir)
    elif args.kendall_rank_mean:
        kendall_rank(args.feature_dict_path, args.output_dir, 'mean', args.fig_title)
    elif args.kendall_rank_std:
        kendall_rank(args.feature_dict_path, args.output_dir, 'std', args.fig_title)
    elif args.retained_token_visualize:
        retained_token_visualize(args.feature_dict_path, args.output_dir)
    elif args.retained_token_analyze:
        retained_token_analyze(args.feature_dict_path, args.output_dir)
    else:
        raise ValueError("Invalid argument")

    