# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from tqdm import tqdm
import pandas as pd

import math


def get_scheduled_keep_rate_list(iters, epoch, shrink_start_epoch, total_epochs,
                       ITERS_PER_EPOCH, base_keep_rate=0.5, max_keep_rate=1, 
                       num_blocks: int = 12,
                       drop_loc: tuple = (3, 6, 9)):
    '''
        Actually, this returns tuple
    '''
    if epoch < shrink_start_epoch:
        return (1.0, ) * num_blocks # override the default keep_rate, do not drop any tokens
    
    if epoch >= total_epochs:
        return None # let the model to follow the default keep_rate
    
    total_iters = ITERS_PER_EPOCH * (total_epochs - shrink_start_epoch)
    iters = iters - ITERS_PER_EPOCH * shrink_start_epoch
    
    target_keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) \
        * (math.cos(iters / total_iters * math.pi) + 1) * 0.5

    keep_rate_list = [1.0] * num_blocks
    for i in range(num_blocks):
        if i in drop_loc:
            keep_rate_list[i] = target_keep_rate
    
    return tuple(keep_rate_list)



def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # dataset specific settings
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    warmup = args.warmup
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    args.loss_fn = loss_fn
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    # 11/30/22: I decouple the dataset and the following hyper-parameters to make it easier to adapt to new datasets
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

    epoch += 1
    # for amp
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        NUM_BLOCKS = 12
        DROP_LOC = eval(args.drop_loc)
        
        if (args.base_keep_rate < 1.0) and (epoch >= args.shrink_start_epoch):
            train_loader.dataset.freqm = 0
            train_loader.dataset.timem = 0

        for i, (audio_input, labels) in enumerate(tqdm(train_loader)):

            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))


            # E-ViT
            it = epoch * len(train_loader)
            ITERS_PER_EPOCH = len(train_loader)
            scheduled_keep_rate_list = get_scheduled_keep_rate_list(it, epoch, shrink_start_epoch=args.shrink_start_epoch,
                                    total_epochs=args.shrink_start_epoch + args.shrink_epochs,
                                    ITERS_PER_EPOCH=ITERS_PER_EPOCH, base_keep_rate=args.base_keep_rate,
                                    num_blocks=NUM_BLOCKS, drop_loc=DROP_LOC)
            
            with autocast():
                audio_output = audio_model(audio_input, keep_rate_list=scheduled_keep_rate_list)
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
                else:
                    loss = loss_fn(audio_output, labels)

            # optimization if amp is not used
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(audio_model, test_loader, args, epoch)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        print('validation finished')

        if epoch >= args.first_eval_epoch:
            if mAP > best_mAP:
                best_mAP = mAP
                if main_metrics == 'mAP':
                    best_epoch = epoch

            if acc > best_acc:
                best_acc = acc
                if main_metrics == 'acc':
                    best_epoch = epoch

            if best_epoch == epoch:
                torch.save(audio_model.state_dict(), f"{args.ramdisk_dir}/models/best_audio_model.pth")

        scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))


        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()


    print('training finished')
    # save best mAP or acc as a text file (dump)
    if main_metrics == 'mAP':
        best_result = [best_epoch, best_mAP]
    else:
        best_result = [best_epoch, best_acc]
    np.savetxt(exp_dir + '/best_result.csv', best_result)

    # Unlike AST, We use the best epoch for evaluation
    os.system(f'cp {args.ramdisk_dir}/models/best_audio_model.pth {args.exp_dir}/models/best_audio_model.pth')
    # clear ramdisk
    os.system(f'rm {args.ramdisk_dir}/models/best_audio_model.pth')



def validate(audio_model, val_loader, args, epoch=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []

    flag_extract_features = args.flag_extract_features
    extract_features_path = args.extract_features_path
    label_csv = args.label_csv


    if flag_extract_features:
        df = pd.read_csv(label_csv)
        index_to_name = dict(zip(df['index'], df['display_name']))

    with torch.no_grad():
        for idx, (audio_input, labels) in tqdm(enumerate(val_loader)):
            audio_input = audio_input.to(device)

            # compute output
            if flag_extract_features:
                audio_output, feature_dict = audio_model(audio_input, flag_extract_features=flag_extract_features)
                audio_output = torch.sigmoid(audio_output)
                argmax_list = torch.argmax(labels, dim=1).tolist()
                str_labels = [index_to_name[index] for index in argmax_list]
                feature_dict['labels'] = str_labels
            else:
                audio_output = audio_model(audio_input)
                if (args.drop_token_blk_idx is not None) and (audio_output == None):
                    # the model returns None because it was not available to discard c1 clusters. don't count this sample
                    continue

                audio_output = torch.sigmoid(audio_output)

            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)
            A_loss.append(0)

            batch_time.update(time.time() - end)
            end = time.time()

            if flag_extract_features:
                for key, value in feature_dict.items():
                    feature_path = f'{extract_features_path}/{key}.{idx:04d}.pth'
                    torch.save(value, feature_path)


        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

    return stats, loss
