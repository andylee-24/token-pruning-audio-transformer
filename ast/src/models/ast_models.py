# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.models.layers import to_2tuple,trunc_normal_
from timm.models.layers import DropPath
from einops import rearrange
import math
from typing import Union


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
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
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

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
        self.num_extra_tokens = 2 # cls_token, dist_token
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

        assert num_left_tokens > 0, f"num_left_tokens should be at least 1"
        # check whether there are enough tokens to select
        topk_idx = None
        if (keep_rate < 1.0):
            assert self.num_extra_tokens == 2
            if custom_rank is None:
                attn_score = attn[:, :, 0, self.num_extra_tokens:].mean(dim=(1))
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
        self.num_extra_tokens = 2


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


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=16, tstride=16, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True,
                 depth=12, audioset_pretrained_model_path: str = None,
                 drop_path_rate=0.0,
                 drop_loc: tuple = None, base_keep_rate: tuple = None):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'
        assert fstride == 16 and tstride == 16, 'Currently only support fstride=16 and tstride=16.'
        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        
        # E-ViT
        timm.models.vision_transformer.Attention = Attention
        timm.models.vision_transformer.Block = Block
        
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
                '''                
                @register_model
                def vit_deit_base_distilled_patch16_384(pretrained=False, **kwargs):
                    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
                    ImageNet-1k weights from https://github.com/facebookresearch/deit.
                    """
                    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
                    model = _create_vision_transformer(
                        'vit_deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
                    return model
                '''
            else:
                raise Exception('We only support base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists(audioset_pretrained_model_path) == False:
                # this model performs 0.4593 mAP on the audioset eval set
                assert False
            sd = torch.load(audioset_pretrained_model_path, map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=16, tstride=16, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size=model_size, verbose=False,
                                   drop_loc=drop_loc, base_keep_rate=base_keep_rate)
            # import pdb; pdb.set_trace()
            
            audio_model = torch.nn.DataParallel(audio_model)
            msg = audio_model.load_state_dict(sd, strict=True)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches

            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            if model_size == 'base384':
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 512, 768).transpose(1, 2).reshape(1, 768, 8, 64)
            else:
                raise ValueError('currently only has base384 and tiny224 AudioSet pretrained model.')
            
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 64:
                new_pos_embed = new_pos_embed[:, :, :, 32 - int(t_dim/2): 32 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            elif t_dim > 64:
                raise ValueError(f'{t_dim=} > 64')
            assert f_dim == 8
            # if f_dim < 12:
            #     new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # # otherwise interpolate
            # elif f_dim > 12:
            #     new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            if model_size == 'base384':
                new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
                
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            
            
            
        # initialize TopK components for attention & blocks
        self.use_custom_rank = None
        self.drop_token_blk_idx = None # not enbled
        self.retain_min = None
        self.retain_max = None

        self.depth = depth
        self.num_extra_tokens = 2
        self.num_heads = 12
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        keep_rate_list = [1.0] * depth
        for drop_loc_idx in drop_loc:
            keep_rate_list[drop_loc_idx] = base_keep_rate

        for blk_id in range(depth):

            # init attn
            self.v.blocks[blk_id].attn.num_extra_tokens = 2
            self.v.blocks[blk_id].attn.block_id = blk_id
            self.v.blocks[blk_id].attn.default_keep_rate = keep_rate_list[blk_id]

            # init block
            self.v.blocks[blk_id].block_id = blk_id
            self.v.blocks[blk_id].num_extra_tokens = 2

            # init ast model
        self.num_extra_tokens = 2
                
            

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x, keep_rate_list: Union[list, tuple, type(None)] = None, 
                flag_extract_features: bool = False):
        
        if (keep_rate_list is not None) and (len(keep_rate_list) != len(self.v.blocks)):
            raise ValueError(f"keep_rate should be a list/tuple of length {len(self.v.blocks)}, got {keep_rate_list}")
            
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        # AST reqquires additional unsqueeze and transpose operations
        # import pdb; pdb.set_trace()
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B, C, F, T = x.shape

        if flag_extract_features == True:
            feature_dict = {'mel': x.cpu()} # (B, 1, T, F)



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
                raise ValueError(f"custom_rank should be in ['mean', 'std'], got {self.use_custom_rank}")
    
        if self.drop_token_blk_idx is not None:
            h = x.size(2) // 16 # (x.shape = B, 3, H, W)
            token_intensity_mean = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=16, q=16, h=h).mean(dim=1)


        x = self.v.patch_embed(x)


        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        

        for blk_idx, blk in enumerate(self.v.blocks):
            keep_rate = keep_rate_list[blk_idx] if (keep_rate_list is not None) else None


                # x = blk(x, keep_rate)

            if flag_extract_features == True:
                x, block_feature_dict = blk(x, keep_rate, flag_extract_features)
                feature_dict.update(block_feature_dict)
            
            elif custom_rank is not None:
                x, topk_idx = blk.forward_with_custom_rank(x, keep_rate, flag_extract_features, custom_rank)
                if topk_idx is not None:
                    custom_rank = torch.gather(custom_rank, dim=1, index=topk_idx)
            
            else:
                x = blk(x, keep_rate, flag_extract_features)

            if self.drop_token_blk_idx == blk_idx: # the default value is -1
                assert B == 1
                # select index of tokens to retain: greater than self.retain_min and smaller than self.retain_max
                retain_idx = torch.nonzero((token_intensity_mean > self.retain_min) & (token_intensity_mean < self.retain_max))
                # keep retain_idx tokens
                if retain_idx[:, 1].shape[0] > 0:
                    x = torch.cat([x[:, 0:self.num_extra_tokens, :], x[:, retain_idx[:, 1] + self.num_extra_tokens, :]], dim=1)
                else:
                    # There is no token to retain, return None
                    return None
                
            
        x = self.v.norm(x)
        # import pdb; pdb.set_trace() -> x.shape == (B, num_tokens, C)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)
        
        if flag_extract_features:
            return x, feature_dict
        else:
            return x
        

if __name__ == '__main__':
    input_tdim = 100
    ast_mdl = ASTModel(input_tdim=input_tdim)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)

    input_tdim = 256
    ast_mdl = ASTModel(input_tdim=input_tdim,label_dim=50, audioset_pretrain=True)
    # input a batch of 10 spectrogram, each with 512 time frames and 128 frequency bins
    # test_input = torch.rand([10, input_tdim, 128])
    # test_output = ast_mdl(test_input)
    # output should be in shape [10, 50], i.e., 10 samples, each with prediction of 50 classes.
    print(test_output.shape)