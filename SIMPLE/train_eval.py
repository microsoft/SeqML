# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import random
import psutil
import numpy as np
from tqdm import tqdm

from utils import AverageMeter, ProgressMeter, accuracy


def train_ensemble(epoch, \
        ensemble_net, data_loader,\
        split = 'train', model_pool = None, args = None):
    
    # Convert network to DataParellel and move it to device
    ensemble_net = torch.nn.DataParallel(ensemble_net).to(args.device)
    # Set the network to training mode
    ensemble_net.train()
    
    # Create AverageMeter objects
    losses = AverageMeter('losses',':6.2f')
    top1 = AverageMeter('top1', ':6.2f')
    top5 = AverageMeter('top5', ':6.2f')

    top1_arch1 = AverageMeter('top1_arch1', ':6.2f')
    ens_improve = AverageMeter('ens_improve', ':6.2f')
    
    # Create ProgressMeter object
    progress = ProgressMeter(
        -1,
        [losses, top1, top5, top1_arch1, ens_improve], prefix=f"Train: ")

    sum_bp_time = 0
    
    # Determine the number of iterations based on the data and the mixed training mode
    if args.super_mixed == False:
        if args.mixed_training == False:
            iterations = ensemble_net.module.steps_per_epoch
        else:
            iterations = 1
    else:
        if epoch % 2 == args.reverse:
            iterations = ensemble_net.module.steps_per_epoch
            data_loader = data_loader[0]
            
        else:
            iterations = 1
            data_loader = data_loader[1]

    # Loop through the dataset
    with tqdm(total=iterations, desc='Iterations') as t1:
        for dataloader_idx in range(iterations):
            # If not super mixed training
            if args.super_mixed == False:
                # If not mixed training
                if args.mixed_training == False:
                    # Fetch minibatches and flatten the tensors
                    minibatches_device = [(x, y, idx, env)
                        for x,y,idx,env in next(data_loader)]
                    x = torch.cat([x for x, y, idx, env in minibatches_device])
                    y = torch.cat([y for x, y, idx, env in minibatches_device])
                    idx = torch.cat([idx for x, y, idx, env in minibatches_device])
                    env = torch.cat([env for x, y, idx, env in minibatches_device])
                    minibatches_device = [(x, y, idx, env)]
                else:
                    # If mixed training
                    minibatches_device = data_loader
            else:
                # If super mixed training
                if epoch % 2 == args.reverse:
                    # Fetch minibatches and shuffle
                    minibatches_device = [(x, y, idx, env)
                        for x,y,idx,env in next(data_loader)]
                    random.shuffle(minibatches_device)
                else:
                    # If even epoch number 
                    minibatches_device = data_loader
            
            with tqdm(total=len(minibatches_device), desc='Train') as t2:
                # Loop through minibatches_device
                for batch_index, (images, labels, idxs, envs) in enumerate(minibatches_device):

                    # If domainbed_dataset is DomainNet and (mixed training and batch_index is greater than split point)
                    if iterations == 1 and batch_index > int(len(minibatches_device)/args.domainnet_split_point) and args.domainbed_dataset == 'DomainNet':
                        break

                    # If ensemble_mode is 'mean', domainbed_dataset is DomainNet and batch_index is greater than or equal to 1
                    if args.ensemble_mode == 'mean' and batch_index >= 1 and args.domainbed_dataset == 'DomainNet':
                        break

                    # Move images and labels to the device
                    images, labels = images.to(args.device), labels.to(args.device)
                    # Call model_pool with images, idxs and envs and get y_hat
                    y_hat = model_pool(images, idxs, envs, epoch = epoch)
                    # Pass images, y_hat, labels, split and epoch to the ensemble_net to obtain
                    output, loss, y_hats, child_loss, confidence_loss, ensemble_loss, bp_time = ensemble_net(images, y_hat, split, labels, epoch = epoch, envs = envs)
                    
                    # Sum up the backward pass time
                    sum_bp_time += bp_time
                    del y_hat

                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    losses.update(loss.item(), labels.size(0))
                    top1.update(acc1[0].item(), labels.size(0))
                    top5.update(acc5[0].item(), labels.size(0))

                    # Calculate top 1 accuracy for each architecture
                    acc1_arch = []
                    for model_idx in range(args.model_num):
                        acc1_arch.append(accuracy(y_hats[model_idx], labels, topk=(1, 5))[0].item())
                    # Get the maximum top 1 accuracy for the 1st architecture
                    acc1_arch1, _ = accuracy(y_hats[0], labels, topk=(1, 5))
                    # Update top1_arch1 object with the highest top 1 accuracy of any architecture
                    top1_arch1.update(np.max(acc1_arch), labels.size(0))
                    # Calculate ensemble improvement by taking the minimum of (top1[0] - top1_arch[x])
                    ensemble_improvement = [acc1[0].item() - item for item in acc1_arch]
                    ensemble_improvement = np.min(ensemble_improvement)

                    ens_improve.update(ensemble_improvement, labels.size(0))
                    del y_hats, images, loss, labels, output, ensemble_loss, child_loss, confidence_loss

                    # Set the postfix message for the inner loop
                    t2.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg,
                        'ensemble_improvement': ens_improve.avg,
                        'top1_arch1': top1_arch1.avg,
                    })
                    t2.update(1)

                    for k, v in progress.compose_json().items():
                        args.writer.add_scalar(f"{k}/ens_train", v, epoch*len(minibatches_device) + batch_index)

                    # Print CPU and memory usage statistics
                    print('cpu_percent:', psutil.cpu_percent(), ' virtual_memory_percent:', psutil.virtual_memory().percent)

    try:
        print('epoch:', epoch, '-- Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (losses.avg, top1.avg, top5.avg))
    except:
        print('epoch:', epoch, '-- Results: loss=%.5f,\t top1=%.1f' % (losses.avg, top1.avg))
    return top1.avg, sum_bp_time


# defining the function to validate ensemble
@torch.no_grad()
def validate_ensemble(epoch,\
        ensemble_net, data_loader,\
        split = 'test', model_pool = None, args = None):

    torch.backends.cudnn.benchmark = True

    ensemble_net = torch.nn.DataParallel(ensemble_net).to(args.device)
    ensemble_net.eval()

    # Create AverageMeters objects for top1 of each architecture
    for model_idx in range(args.model_num):
        globals()['top1_arch' + str(model_idx+1)] = AverageMeter('top1_arch' + str(model_idx+1), ':6.2f')

    # Creating objects for variables to keep track of ensemble improvement
    ens_improve = AverageMeter('ens_improve', ':6.2f')

    # Creating objects for AverageMeters for losses, top1 and top5 accuracy
    losses = AverageMeter('losses',':6.2f')
    top1 = AverageMeter('top1', ':6.2f')
    top5 = AverageMeter('top5', ':6.2f')

    # Creating the ProgressMeter object
    progress = ProgressMeter(
        -1,
        [losses, top1, top5, ens_improve] + [globals()['top1_arch' + str(model_idx+1)] for model_idx in range(args.model_num)],
        prefix=f"Eval: ",
        logger=None)

    # Checking for correct data_loader type
    if type(data_loader) != list:
        data_loader = [data_loader]
    
    # Looping through each data loader
    for dataloader_idx in range(len(data_loader)):
        # Looping through each batch of images, labels, idx and envs and keeping track of progress using tqdm
        with tqdm(total=len(data_loader[dataloader_idx]), desc='Val') as t:
            for batch_index, (images, labels, idxs, envs) in enumerate(data_loader[dataloader_idx]):

                # Breaking the loop at center of validation dataset for DomainNet
                if split == 'validation' and batch_index > int(len(data_loader[dataloader_idx])/2) and args.domainbed_dataset == 'DomainNet':
                    break

                images, labels = images.to(args.device), labels.to(args.device)
                y_hat = model_pool(images, idxs, envs)
                output, loss, y_hats, child_loss, confidence_loss, ensemble_loss = ensemble_net.module.predict(images, y_hat, split, labels, args = args, ensemble_mode = args.ensemble_mode, batch_index = batch_index)
                del y_hat
                
                # Calculating accuracy scores
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                losses.update(loss.item(), labels.size(0))
                top1.update(acc1[0].item(), labels.size(0))
                top5.update(acc5[0].item(), labels.size(0))

                # Calculating accuracy scores for each individual model in the ensemble
                acc1_arch = []
                for model_idx in range(args.model_num):
                    acc1_arch.append(accuracy(y_hats[model_idx], labels, topk=(1, 5))[0].item())

                for model_idx in range(args.model_num):
                    eval('top1_arch' + str(model_idx+1)).update(acc1_arch[model_idx], labels.size(0))

                # Calculating the ensemble improvement
                ensemble_improvement = [acc1[0].item() - item for item in acc1_arch]
                ensemble_improvement = np.min(ensemble_improvement)

                ens_improve.update(ensemble_improvement, labels.size(0))
                del y_hats, images, loss, labels, output, ensemble_loss, child_loss, confidence_loss

                # Updating the tqdm progress bar
                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg,
                    'ensemble_improvement': ens_improve.avg,
                })
                t.update(1)

                # Updating the tensorboard metrics
                for k, v in progress.compose_json().items():
                    args.writer.add_scalar(f"{k}/ens_train", v, dataloader_idx*len(data_loader[dataloader_idx]) + batch_index)

    # Printing the final results
    print('epoch:', epoch, '-- Results: loss=%.5f,\t top1=%.1f' % (losses.avg, top1.avg))
    print('individual results:')

    # Printing the individual results for each model in the ensemble
    ranked_list = []
    for model_idx in range(args.model_num):
        try:
            print(args.pretrained_models[model_idx], ': ', eval('top1_arch' + str(model_idx+1)).avg)
        except:
            print(eval('top1_arch' + str(model_idx+1)).avg)

        ranked_list.append(eval('top1_arch' + str(model_idx+1)).avg)
    
    # Using topk to get the sorted indices in descending order
    ranked_list = torch.tensor(ranked_list)
    topk_index = torch.topk(ranked_list, args.rough_rank_num if args.rough_rank_num != -1 and args.rough_rank_num > len(ranked_list) else len(ranked_list)).indices

    print('topk_index:', topk_index)
    print('Test epoch:', epoch, '-- Results: loss=%.5f,\t top1=%.1f' % (losses.avg, top1.avg))
    print('individual results:')
    print([str(model_pool.model_name[model_idx]) + ': ' + str(eval('top1_arch' + str(model_idx+1)).avg) for model_idx in range(args.model_num)])
    
    # returning top1 average score and topk index positions
    return top1.avg, topk_index

