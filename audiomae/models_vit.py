# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
from typing import Union

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# import DropPath, to_2tuple, trunc_normal_ from timm packages
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


# Attention -> Block -> VisionTransformer


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 block_id: int = 0,
                 default_keep_rate: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_extra_tokens = 1 # cls token (unused, mean_pooling == True)
        self.block_id = block_id
        self.default_keep_rate = default_keep_rate
        assert 0.0 < self.default_keep_rate <= 1.0, f"default_keep_rate should be in (0, 1], got {self.default_keep_rate}"

    def forward(self, x, keep_rate = None, flag_extract_features = False, custom_rank = None, attn_mask = None):
        '''
        extract_features: Q, K, V, topk_idx
        returns 
            x, topk_idx or
            x, topk_idx, attn_feature_dict
        '''
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            # import pdb; pdb.set_trace()
            # attn = attn * attn_mask
            # use modified softmax
            # https://github.com/raoyongming/DynamicViT/blob/1322e626d1e9eca18cc150569908fb4af29e15f7/models/dyvit.py#L168
            eps = 1e-6
            max_attn = torch.max(attn, dim=-1, keepdim=True)[0]
            attn = (attn - max_attn).to(torch.float32).exp_() * attn_mask
            attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
            attn = attn.type_as(v)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

                
        # Top-K selection begin
        if keep_rate is None:
            keep_rate = self.default_keep_rate
            
        num_left_tokens = math.ceil(keep_rate * (N - self.num_extra_tokens))
        
        assert num_left_tokens > 0
            # raise ValueError(f"num_left_tokens should be at least 2, got {num_left_tokens}")
        # check whether there are enough tokens to select
        topk_idx = None
        if (keep_rate < 1.0):

            if custom_rank is None:
                attn_score = attn[:, :, self.num_extra_tokens:, self.num_extra_tokens:].mean(dim=(1, 2))
                _, topk_idx = torch.topk(attn_score, num_left_tokens, dim=1, largest=True, sorted=True)
            else:
                _, topk_idx = torch.topk(custom_rank, num_left_tokens, dim=1, largest=True, sorted=True)
                
        # Top-K selection end
        
        
        # For simplicity, class Attention does not have def forward_with_extract_features
        if flag_extract_features: 
            
            # attn_feature_dict = {f'block-{self.block_id}.Q': q.cpu(), 
            #                      f'block-{self.block_id}.K': k.cpu(), 
            #                      f'block-{self.block_id}.V': v.cpu(),
            #                      f'block-{self.block_id}.attn_score': attn[:, :, self.num_extra_tokens:, self.num_extra_tokens:].mean(dim=(1, 2))}
            
            attn_feature_dict = {f'block-{self.block_id}.attn_score': attn[:, :, self.num_extra_tokens:, self.num_extra_tokens:].mean(dim=(1, 2)).cpu()}
            
            if topk_idx is not None:
                attn_feature_dict[f'block-{self.block_id}.topk_idx'] = topk_idx.cpu()
            return x, topk_idx, attn_feature_dict
        else:
            return x, topk_idx


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 block_id: int = 0,
                 default_keep_rate: float = 1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, block_id=block_id, default_keep_rate=default_keep_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.block_id = block_id
        self.num_extra_tokens = 1

    def forward_with_extract_features(self, x, keep_rate = None):
        '''
        extract_feautres:
            attn: Q, K, V, topk_idx
        '''
        
        x_input = x
        
        norm1_output = self.norm1(x)
        attn_output, topk_idx, attn_feature_dict = self.attn(norm1_output, keep_rate=keep_rate, flag_extract_features=True)
        
        drop_path_output = self.drop_path(attn_output)
        x = x + drop_path_output
        
        if topk_idx is not None:
            topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            topk_non_extra_tokens = torch.gather(x[:, self.num_extra_tokens:], dim=1, index=topk_idx_expanded)
            x = torch.cat([x[:, 0:self.num_extra_tokens], topk_non_extra_tokens], dim=1)
                        
        norm2_output = self.norm2(x)
        mlp_output = self.mlp(norm2_output)
        drop_path_output_2 = self.drop_path(mlp_output)
        x = x + drop_path_output_2
                
        block_feature_dict = {
            # f'block-{self.block_id}.x_input': x_input.cpu(), 
            # f'block-{self.block_id}.norm1_output': norm1_output.cpu(), 
            # f'block-{self.block_id}.attn_output': attn_output.cpu(),
            # f'block-{self.block_id}.mlp_output': mlp_output.cpu(),
        }
        
        block_feature_dict.update(attn_feature_dict)
        
        return x, block_feature_dict
    
    def forward(self, x, keep_rate = None, flag_extract_features:bool = False, attn_mask = None):
        
        if flag_extract_features:
            # returns x, block_feature_dict
            return self.forward_with_extract_features(x, keep_rate)
        
        attn_output, topk_idx = self.attn(self.norm1(x), keep_rate, flag_extract_features, attn_mask=attn_mask)
        x = x + self.drop_path(attn_output)
        
        if topk_idx is not None:
            topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            topk_non_extra_tokens = torch.gather(x[:, self.num_extra_tokens:], dim=1, index=topk_idx_expanded)
            x = torch.cat([x[:, 0:self.num_extra_tokens], topk_non_extra_tokens], dim=1)
                
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

    def forward_with_custom_rank(self, x, keep_rate = None, flag_extract_features:bool = False, custom_rank = None):
        
        if flag_extract_features:
            # returns x, block_feature_dict
            return self.forward_with_extract_features(x, keep_rate)
        
        attn_output, topk_idx = self.attn(self.norm1(x), keep_rate, flag_extract_features, custom_rank)
        x = x + self.drop_path(attn_output)
        
        if topk_idx is not None:
            topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.gather(x, dim=1, index=topk_idx_expanded)
                
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x, topk_idx
    

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


 
    
    
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 mean_pooling=False, mask_2d=True, target_length=None, 
                 drop_loc: tuple = None, base_keep_rate: tuple = None, **kwargs):
        
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.num_extra_tokens = 1
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        keep_rate_list = [1.0] * depth
        for idx, drop_loc_idx in enumerate(drop_loc):
            keep_rate_list[drop_loc_idx] = base_keep_rate
            
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                block_id=i,
                default_keep_rate=keep_rate_list[i])
            for i in range(depth)])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
        # AudioMAE parameters
        assert mean_pooling == True
        self.mean_pooling = mean_pooling
        if self.mean_pooling:
            self.fc_norm = norm_layer(embed_dim)
        # del self.norm  # remove the original norm
        self.mask_2d = mask_2d
        self.target_length = target_length
        
        self.use_custom_rank = None
        self.retain_max = None
        self.retain_min = None
        self.drop_token_blk_idx = None


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def forward_features(self, x, keep_rate_list = None, flag_extract_features: bool = False):
        B, _, T, F = x.shape
        assert T >= F and F == 128
                
        if flag_extract_features == True:
            feature_dict = {'mel': x.cpu()}
            
        custom_rank = None # for ablation study
        # import pdb; pdb.set_trace()
        if self.use_custom_rank is not None:
            assert flag_extract_features == False
            h = x.size(2) // 16 # (x.shape = B, 3, H, W)
            if self.use_custom_rank == 'mean':
                custom_rank = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h).mean(dim=1)
            elif self.use_custom_rank == 'std':
                custom_rank = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h).std(dim=1)
            else:
                assert False, f"Unknown use_custom_rank: {self.use_custom_rank}"
            
        if self.drop_token_blk_idx is not None:
            h = x.size(2) // 16 # (x.shape = B, 3, H, W)
            token_intensity_mean = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h).mean(dim=1)
                        
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        
        for idx, blk in enumerate(self.blocks):
            keep_rate = keep_rate_list[idx] if (keep_rate_list is not None) else None

            if flag_extract_features == True:
                x, block_feature_dict = blk(x, keep_rate, flag_extract_features)
                feature_dict.update(block_feature_dict)
            elif custom_rank is not None:
                x, topk_idx = blk.forward_with_custom_rank(x, keep_rate, flag_extract_features, custom_rank)
                if topk_idx is not None:
                    custom_rank = torch.gather(custom_rank, dim=1, index=topk_idx)
            else:
                x = blk(x, keep_rate, flag_extract_features)

            if self.drop_token_blk_idx == idx: # the default value is -1
                assert B == 1
                retain_idx = torch.nonzero((token_intensity_mean > self.retain_min) & (token_intensity_mean < self.retain_max))
                # keep retain_idx tokens
                if retain_idx[:, 1].shape[0] > 0:
                    x = torch.cat([x[:, 0:self.num_extra_tokens, :], x[:, retain_idx[:, 1] + self.num_extra_tokens, :]], dim=1)
                else:
                    return None

        if self.mean_pooling:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            assert False # no cls token

        if flag_extract_features:
            return outcome, feature_dict
        else:
            return outcome

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        N, L, D = x.shape  # batch, length, dim

        T = (self.target_length // 16)
        F = 8 # (128 // 16)
    
        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0,2,1,3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None


    def forward_features_mask(self, x, keep_rate_list=None, 
                              mask_t_prob:float = 1.0, mask_f_prob:float = 1.0):
        B = x.shape[0] #4,1,1024,128
        x = self.patch_embed(x) # 4, 512, 768

        x = x + self.pos_embed[:, 1:, :]
        if self.random_masking_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_t_prob)
            
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)        
        
        x = self.pos_drop(x)

        # apply Transformer blocks
        for idx, blk in enumerate(self.blocks):
            keep_rate = keep_rate_list[idx] if (keep_rate_list is not None) else None
            x = blk(x, keep_rate=keep_rate, flag_extract_features=False)

        if self.mean_pooling:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            assert False # no cls token


        return outcome



    # overwrite original timm
    def forward(self, x, keep_rate_list: Union[list, tuple, type(None)] = None, 
                mask_t_prob=0.0, mask_f_prob=0.0, 
                flag_extract_features: bool = False):
        
        if (keep_rate_list is not None) and (len(keep_rate_list) != len(self.blocks)):
            raise ValueError(f"keep_rate should be a list/tuple of length {len(self.blocks)}, got {keep_rate_list}")
        
        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            assert flag_extract_features == False # We don't both train and extract features
            x = self.forward_features_mask(x, keep_rate_list=keep_rate_list, 
                                           mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)
            
        elif flag_extract_features == True:
            x, feature_dict = self.forward_features(x, keep_rate_list, flag_extract_features=True)
        else:
            x = self.forward_features(x, keep_rate_list, flag_extract_features=False)
        
        if x is None:
            return None
        
        x = self.head(x)
        
        if flag_extract_features:
            return x, feature_dict
        else:
            return x



def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)        
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
