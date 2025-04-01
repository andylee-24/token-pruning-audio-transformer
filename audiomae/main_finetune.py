# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import yaml
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from termcolor import colored

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed_audio
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.sampler import DistributedEvalSampler

import models_vit

from engine_finetune import train_one_epoch
from dataset import AudiosetDataset, VoxCeleb1Dataset
from timm.models.vision_transformer import PatchEmbed

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', required=True, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.0)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')


    # EMA related parameters
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--audioset_pretrained_model_path', required=True,
                        help='audioset_pretrained_model_path from checkpoint')
    parser.add_argument('--finetuned_model_path', default='', help='finetune_model_path from checkpoint')
    parser.add_argument('--mean_pooling', type=str2bool, required=True)

    # Dataset parameters
    parser.add_argument('--voxceleb1_root', type=str, default=None)
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', required=True, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # training parameters
    # saving best checkpoint to the ramdisk avoid unnecessary disk write.
    parser.add_argument('--ramdisk_dir', type=str, required=True)
    parser.add_argument('--first_eval_ep', default=0, type=int, help='do eval after first_eval_ep')

    # For audioset
    parser.add_argument('--audio_exp', action='store_true', help='audio exp')
    parser.add_argument("--data_train", type=str, default='', help="training data json")
    parser.add_argument("--data_eval", type=str, default='', help="validation data json")
    parser.add_argument("--label_csv", type=str, default='', help="csv with class labels")
    parser.add_argument('--freqm', help='frequency mask max length', type=int, required=True)
    parser.add_argument('--timem', help='time mask max length', type=int, required=True)    
    parser.add_argument('--mask_2d', type=str2bool, required=True)
    parser.add_argument('--roll_mag_aug', type=str2bool, default=False, help='use roll_mag_aug')
    parser.add_argument('--mask_t_prob', default=0.0, type=float, help='T masking ratio (percentage of removed patches).') #  
    parser.add_argument('--mask_f_prob', default=0.0, type=float, help='F masking ratio (percentage of removed patches).') #  
    parser.add_argument("--dataset", type=str, required=True, help="the dataset used", choices=["audioset", "esc50", "spc2", "voxceleb1"])
    parser.set_defaults(audio_exp=True)
    
    # debug parameters
    parser.add_argument('--flag_extract_features', type=str2bool, default=False)
    parser.add_argument('--extract_features_path', type=str, default=None)
    
    # pruning parameters
    parser.add_argument('--drop_loc', default='(3, 6, 9)', type=str, 
                        help='the layer indices for shrinking inattentive tokens')
    parser.add_argument('--base_keep_rate', type=float, default=1.0,
                        help='Base keep rate (default: 1.0)')
    parser.add_argument('--shrink_epochs', default=0, type=int, 
                        help='how many epochs to perform gradual shrinking of inattentive tokens')
    parser.add_argument('--shrink_start_epoch', default=10, type=int, 
                        help='on which epoch to start shrinking of inattentive tokens')
    
    parser.add_argument('--custom_rank', default=None, type=str, help='custom rank ablation study')

    # ablation study - discard low intensity tokens
    parser.add_argument('--retain_min', default=-100, type=float)
    parser.add_argument('--retain_max', default=100, type=float)
    parser.add_argument('--drop_token_blk_idx', type=int)
    parser.add_argument('--result_path', type=str, default=None)
    
    return parser

def args_checker(args):
    
    if args.eval is False:
        # training mode
        assert args.flag_extract_features == False, 'extract_features is only supported during evaluation'
    else:
        assert args.finetuned_model_path, 'finetuned_model_path is required for evaluation'
    
    if args.flag_extract_features:
        assert args.eval, 'flag_extract_features is only supported during evaluation'
        assert args.finetuned_model_path, 'finetuned_model_path is required for feature extraction'
        assert args.extract_features_path, 'extract_features_path is required for feature extraction'
        assert misc.get_world_size() == 1, 'feature extraction only support single GPU'
        
        
def main(args):
    misc.init_distributed_mode(args)
    args_checker(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
        
    if not args.audio_exp:
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)
    else:
        norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'spc2':[-6.845978, 5.5654526], 'voxceleb1': [-6.370, 3.074]}
        target_length = {'audioset':1024, 'esc50':512, 'spc2':128, 'voxceleb1': 1024}
        multilabel_dataset = {'audioset': True, 'esc50': False, 'spc2': True, 'voxceleb1': False}
        use_noise = {'audioset': False, 'esc50': False, 'spc2': True, 'voxceleb1': True}
        loss_fn_type = {'audioset': 'bce', 'esc50': 'ce', 'spc2': 'bce', 'voxceleb1': 'ce'}
        audio_conf_train = {'num_mel_bins': 128, 
                      'target_length': target_length[args.dataset], 
                      'freqm': args.freqm,
                      'timem': args.timem,
                      'mixup': args.mixup,
                      'dataset': args.dataset,
                      'mode':'train',
                      'mean':norm_stats[args.dataset][0],
                      'std':norm_stats[args.dataset][1],
                      'multilabel':multilabel_dataset[args.dataset],
                      'noise':use_noise[args.dataset]}
        audio_conf_val = {'num_mel_bins': 128, 
                      'target_length': target_length[args.dataset], 
                      'freqm': 0,
                      'timem': 0,
                      'mixup': 0,
                      'dataset': args.dataset,
                      'mode':'val',
                      'mean':norm_stats[args.dataset][0],
                      'std':norm_stats[args.dataset][1],
                      'multilabel':multilabel_dataset[args.dataset],
                      'noise':False}
        
        if args.dataset == 'voxceleb1':                  
            dataset_train = VoxCeleb1Dataset(root=args.voxceleb1_root, subset='train', target_length=target_length[args.dataset], audio_conf=audio_conf_train)
            dataset_val = VoxCeleb1Dataset(root=args.voxceleb1_root, subset='test', target_length=target_length[args.dataset], audio_conf=audio_conf_val)
        else:
            dataset_train = AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf_train, roll_mag_aug=args.roll_mag_aug)
            dataset_val = AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=audio_conf_val)

    if True: #args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            # sampler_val = torch.utils.data.DistributedSampler(
            #     dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_val = DistributedEvalSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if misc.is_main_process() and (not args.eval):
        log_writer_path = os.path.join(args.output_dir, 'tb_log')
        # if log_writer_path exists, stop training
        if os.path.exists(log_writer_path):
            print(colored(f'!! path {log_writer_path} exists, stop training', 'yellow'))
            exit(1)
        log_writer = SummaryWriter(log_writer_path)
    else:
        log_writer = None
        
    if misc.is_main_process() and (not args.eval):
        with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)     

    torch.distributed.barrier()
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )

    if args.drop_token_blk_idx is not None:
        assert args.eval == True and args.base_keep_rate == 1.0
        val_batch_size=1
    else:
        val_batch_size=args.batch_size
    

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=val_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        # mixup_fn = Mixup(...) AudioMAE does mixup in the dataset
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        mean_pooling=args.mean_pooling,
        mask_2d=args.mask_2d,
        target_length=target_length[args.dataset],
        drop_loc=eval(args.drop_loc),
        base_keep_rate=args.base_keep_rate,
    )
    
    if args.model == 'vit_base_patch16':
        hidden_dim_size=768
    elif args.model == 'vit_small_patch16':
        hidden_dim_size=384
    
    
    if args.audio_exp:
        img_size=(target_length[args.dataset],128) # 1024, 128
        in_chans=1

        model.patch_embed = PatchEmbed(img_size, 16, in_chans, hidden_dim_size) # no overlap. stride=img_size=16
        # num_patches = model.patch_embed.num_patches
        # num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        num_patches = (128 // 16) * (target_length[args.dataset] // 16)
        model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim_size), requires_grad=False)  # fixed sin-cos embedding

    checkpoint = torch.load(args.audioset_pretrained_model_path, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % args.audioset_pretrained_model_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]


    if not args.audio_exp:
        assert False
        interpolate_pos_embed(model, checkpoint_model)	
    else: # override for audio_exp for now	
        # imgnet: 14,14	
        # audioset size: (8,64)	
        # esc size: (8,32)	
        # only when pt-ft mismatch when loading an imgnet-pt model	
        #interpolate_patch_embed_audio(model, checkpoint_model, orig_channel=3, new_channel=1)	
        # when imgnet-pt(224/16,224/16) to audioset(128//16,1024//16)	
        # imgnet-pt	
        #interpolate_pos_embed_audio(model, checkpoint_model, orig_size=(14,14), new_size=(8,64))	
        
        new_size=(128 // 16, target_length[args.dataset] // 16)
        interpolate_pos_embed_audio(model, checkpoint_model, orig_size=(8,64), new_size=new_size)	
        # when audioset-pt(128/16,1024/16) to audioset-ft(128//10,1024//10) (change to overlap) # try avoiding this	
        #interpolate_pos_embed_audio(model, checkpoint_model, orig_size=(8,64), new_size=(12,101)) 	

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    #if args.mean_pooling:
    #    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    #else:
    #    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    if args.finetuned_model_path:
        finetuned_checkpoint_model = torch.load(args.finetuned_model_path, map_location='cpu')
        msg = model.load_state_dict(finetuned_checkpoint_model['model'], strict=True)            
            
        
    model.to(device)
        
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.custom_rank is not None:
        model.use_custom_rank = args.custom_rank
        print(colored(f"Using {args.custom_rank} instead of attention score", "green"))
    
    if args.drop_token_blk_idx is not None:
        model.retain_min = args.retain_min
        model.retain_max = args.retain_max
        model.drop_token_blk_idx = args.drop_token_blk_idx
        print(colored(f"ablation study: {args.retain_min=}, {args.retain_max=}, discard loc {args.drop_token_blk_idx}", "green"))
        
        
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))


    loss_scaler = NativeScaler()

    if loss_fn_type[args.dataset] == 'bce':
        # smoothing is handled with mixup label transform
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss_fn_type[args.dataset] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Invalid loss_fn_type: {loss_fn_type[args.dataset]}')

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    if args.dataset == 'audioset':
        metric_type = 'mAP'
        from engine_finetune import evaluate_audioset as evaluate
    else:
        metric_type = 'acc1'
        from engine_finetune import evaluate
        
    if args.eval:
        if args.extract_features_path:
            Path(args.extract_features_path).mkdir(parents=True, exist_ok=False)
            
        test_stats = evaluate(data_loader_val, model, device, flag_extract_features=args.flag_extract_features, extract_features_path=args.extract_features_path,
                              label_csv=args.label_csv)
        print(f"{metric_type} of the network on the {len(dataset_val)} test dataset: {test_stats[metric_type]:.4f}")
        if args.result_path is not None:
            with open(args.result_path, 'w') as f:
                f.write(f"{test_stats[metric_type]:.4f}")
        exit(0)
        
    assert args.flag_extract_features == False # not support feature extraction during training

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = 0.0
    best_score_epoch = 0
    
    assert args.mask_t_prob == args.mask_f_prob
    mask_prob = args.mask_t_prob
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        if (args.base_keep_rate < 1.0) and (epoch >= args.shrink_start_epoch):
            # disable all augmentations
            mask_prob = 0.0
            data_loader_train.dataset.freqm = 0
            data_loader_train.dataset.timem = 0
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            mask_prob=mask_prob,
            args=args
        )
        
        if args.dataset == 'audioset': # eval dataset is huge
            if epoch >= args.first_eval_ep:
                test_stats = evaluate(data_loader_val, model, device)
            else:
                print('skip eval')
                test_stats = {'mAP': -1.0}
        else:
            if epoch >= args.first_eval_ep:
                test_stats = evaluate(data_loader_val, model, device)
            else:
                print('skip eval')
                test_stats = {'acc1': -1.0}
            
        print(f"{metric_type} of the network on the {len(dataset_val)} test data: {test_stats[metric_type]:.1f}%")
        
        if max_score <= test_stats[metric_type]:
            max_score = test_stats[metric_type]
            best_score_epoch = epoch
            if args.output_dir and misc.is_main_process():
                # remove other checkpoints in ramdisk
                os.system(f'rm {args.ramdisk_dir}/checkpoint-*.pth')
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, use_ramdisk=True)
                
            torch.distributed.barrier()


        print(f'Max {metric_type}: {max_score:.4f}%')    

        if log_writer is not None:
            for k, v in train_stats.items():
                log_writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in test_stats.items():
                log_writer.add_scalar(f'test/{k}', v, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.output_dir and misc.is_main_process():
        best_score_epoch_str = f'{best_score_epoch:03}'
        f = Path(args.output_dir) / f'best-{best_score_epoch_str}-{max_score:.4f}.txt'
        f.touch()
        os.system(f'cp {args.ramdisk_dir}/checkpoint-{best_score_epoch_str}.pth {args.output_dir}/best_model.pth')
        # clear ramdisk
        os.system(f'rm {args.ramdisk_dir}/checkpoint-*.pth')
    
    if (args.result_path is not None) and misc.is_main_process():
        with open(args.result_path, 'w') as f:
            f.write(f"{max_score:.4f}")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
