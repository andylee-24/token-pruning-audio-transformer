# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import yaml
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from termcolor import colored

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument("--ramdisk_dir", type=str, required=True)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=16, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=16, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

# EViT params
parser.add_argument('--shrink_start_epoch', type=int, default=999)
parser.add_argument('--shrink_epochs', type=int, default=999)
parser.add_argument('--base_keep_rate', type=float, default=1.0)
parser.add_argument('--drop_loc', default='(3, 6, 9)', type=str, help='the layer indices for shrinking inattentive tokens')

parser.add_argument('--extract_features_path', type=str, default=None)
parser.add_argument('--flag_extract_features', type=str2bool, default=False)


parser.add_argument('--audioset_pretrained_model_path', type=str, default=None, help='Path to the pretrained model')
parser.add_argument('--eval', action='store_true', help='Skip training')
parser.add_argument('--model_size', type=str, default='base384', help='Model size')
parser.add_argument('--seed', type=int, default=0, help='Random seed for esc-50')

    
parser.add_argument('--custom_rank', default=None, type=str, help='custom rank ablation study')
parser.add_argument('--retain_min', default=-100.0, type=float)
parser.add_argument('--retain_max', default=100.0, type=float)
parser.add_argument('--drop_token_blk_idx', default=None, type=int)
parser.add_argument('--first_eval_epoch', default=0, type=int)
parser.add_argument('--eval_result_path', default='eval_result.csv', type=str)


# if args.dataset == 'audioset':
#     if len(train_loader.dataset) > 2e5:
#         print('scheduler for full audioset is used')
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5], gamma=0.5, last_epoch=-1)
#     else:
#         print('scheduler for balanced audioset is used')
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25], gamma=0.5, last_epoch=-1)
#     main_metrics = 'mAP'
#     loss_fn = nn.BCEWithLogitsLoss()
#     warmup = True
# elif args.dataset == 'esc50':
#     print('scheduler for esc-50 is used')
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
#     main_metrics = 'acc'
#     loss_fn = nn.CrossEntropyLoss()
#     warmup = False
# elif args.dataset == 'speechcommands':
#     print('scheduler for speech commands is used')
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
#     main_metrics = 'acc'
#     loss_fn = nn.BCEWithLogitsLoss()
#     warmup = False
# else:
#     raise ValueError('unknown dataset, dataset should be in [audioset, speechcommands, esc50]')
#

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# transformer based model
if args.model == 'ast':
    print('now train a audio spectrogram transformer model')

    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
    # dataset spectrogram mean and std, used to normalize the input
    # norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
    # target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
    # # if add noise for data augmentation, only use for speech commands
    # noise = {'audioset': False, 'esc50': False, 'speechcommands':True}


    timem_conf = {
        'audioset': 192, 'speechcommands': 48, 'esc50': 96
    }
    freqm_conf = {
        'audioset': 48, 'speechcommands': 48, 'esc50': 24
    }
    
    timem = timem_conf[args.dataset]
    freqm = freqm_conf[args.dataset]

    if args.drop_token_blk_idx is not None:
        assert args.base_keep_rate == 1.0 and args.eval == True
        print(colored(f"ablation study: we force the batch size as 1", "red"))
        eval_batch_size = 1
    else:
        eval_batch_size = args.batch_size * 2

    audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': freqm, 'timem': timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                  'noise':args.noise}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}
    
    if args.bal == 'bal':
        print('balanced sampler is being used')
        assert False
        # samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        # sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        # train_loader = torch.utils.data.DataLoader(
        #     dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        #     batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    '''
        def __init__(self, label_dim=527, fstride=16, tstride=16, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True,
                 depth=12, audioset_pretrained_model_path: str = None,
                 drop_path_rate=0.0,
                 drop_loc: tuple = None, base_keep_rate: tuple = None):
    '''

    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size=args.model_size,
                                  depth=12, audioset_pretrained_model_path=args.audioset_pretrained_model_path,
                                  drop_loc=eval(args.drop_loc), base_keep_rate=args.base_keep_rate) # args.signal_ratio <- not during training
    

    if args.custom_rank is not None:
        audio_model.use_custom_rank = args.custom_rank
        print(colored(f"Using {args.custom_rank} instead of attention score", "green"))
    
    if args.drop_token_blk_idx is not None:
        audio_model.retain_min = args.retain_min
        audio_model.retain_max = args.retain_max
        audio_model.drop_token_blk_idx = args.drop_token_blk_idx
        print(colored(f"ablation study: discard low inensity clusters"))


print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir, exist_ok=True)
os.makedirs("%s/models" % args.ramdisk_dir, exist_ok=True)

device = torch.device(0)

if args.eval == False:

    with open("%s/args.yaml" % args.exp_dir, "w") as f:
        yaml.dump(vars(args), f)

    print('Now starting training for {:d} epochs'.format(args.n_epochs))
    train(audio_model, train_loader, val_loader, args)

    # for speechcommands dataset, evaluate the best model on validation set on the test set
    if args.dataset == 'speechcommands':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
        audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd)

        # best model on the validation set
        stats, _ = validate(audio_model, val_loader, args, 'valid_set')
        # note it is NOT mean of class-wise accuracy
        val_acc = stats[0]['acc']
        val_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the validation set---------------')
        print("Accuracy: {:.6f}".format(val_acc))
        print("AUC: {:.6f}".format(val_mAUC))

        # test the model on the evaluation set
        eval_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
        eval_acc = stats[0]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the test set---------------')
        print("Accuracy: {:.6f}".format(eval_acc))
        print("AUC: {:.6f}".format(eval_mAUC))
        np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
else:
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device))
    audio_model.eval()
    
    if args.flag_extract_features:
        os.makedirs(args.extract_features_path, exist_ok=False)


    if args.dataset == 'speechcommands':
        eval_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, _ = validate(val_loader=eval_loader, audio_model=audio_model, args=args)
    else:
        stats, _ = validate(val_loader=val_loader, audio_model=audio_model, args=args)

    # if main metric is accuracy, we use the first element of the stats list
    if args.metrics == 'acc':
        score = stats[0]['acc']
        print("acc: {:.6f}".format(score))
    else:
        # calculate mAP
        score = np.mean([stat['AP'] for stat in stats])
        print("mAP: {:.6f}".format(score))

    # save the evaluation result as a eval-result.csv
    np.savetxt(os.path.join(args.exp_dir, args.eval_result_path), [-1, score])