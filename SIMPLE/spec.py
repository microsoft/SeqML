# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# LOAD all the needed package
import os
import sys
import ssl
import time
import json
import torch
import random
import argparse
import numpy as np
from tensorboardX import SummaryWriter

from utils import Logger
from train_eval import validate_ensemble, train_ensemble
from ensemble import Model_pool, Ensemble_of_ensemble
from save_inference import inference_saver

#suppress SSL certificate verification errors for https requests, which may occur when downloading data
sys.path.append('..')
ssl._create_default_https_context = ssl._create_unverified_context


# parse some configs from the common line
parser = argparse.ArgumentParser()
parser.description=''

#declare and specify the command-line arguments that the script accepts
parser.add_argument("--seed", "--seed", help="this is random seed", dest="random_seed", type=int, default=42)
parser.add_argument("--dataset", "--dataset", help="this is dataset that will be tested on", dest="dataset", type=str, default='domainbed')
parser.add_argument("--domainbed_dataset", "--domainbed_dataset", dest="domainbed_dataset", default=None, type=str)
parser.add_argument("--domainbed_test_env", "--domainbed_test_env", dest="domainbed_test_env", default=0, type=int)
parser.add_argument("--pretrain_model_list", "--pretrain_model_list", dest="pretrain_model_list", default='modelpool_list/model_list_test.txt', type=str)

# mode setting
parser.add_argument("--save_inference_only", "--save_inference_only", dest="save_inference_only", action='store_true')
parser.add_argument("--pure_imagenet_pretrained", "--pure_imagenet_pretrained", dest="pure_imagenet_pretrained", action='store_true')
parser.add_argument("--idx_for_inference_saving", "--idx_for_inference_saving", dest="idx_for_infer_save", default=-1, type=int)
parser.add_argument("--model_idx", "--model_idx", help="this is model_idx (select one model to see its performance)", dest="model_idx", type=int, default=None)

# settings for clfs or matching network
parser.add_argument("--use_true_conf", "--use_true_conf", dest="use_true_conf", action='store_true')
parser.add_argument("--use_optimal", "--use_optimal", dest="use_optimal", action='store_true')
parser.add_argument("--mixed_training", "--mixed_training", dest="mixed_training", action='store_true')
parser.add_argument("--average_ensemble", "--average_ensemble", dest="average_ensemble", action='store_true')
parser.add_argument("--average_clf", "--average_clf", dest="average_clf", action='store_true')
parser.add_argument("--start_iteration_ensemble", "--start_iteration_ensemble", dest="start_iteration_ensemble", default=100, type=int)
parser.add_argument("--start_iteration_clf", "--start_iteration_clf", dest="start_iteration_clf", default=100, type=int)
parser.add_argument("--detach_child", "--detach_child", dest="detach_child", action='store_true')
parser.add_argument("--no_local_weighting", "--no_local_weighting", dest="no_local_weighting", action='store_true')
parser.add_argument("--super_mixed", "--super_mixed", dest="super_mixed", action='store_true')
parser.add_argument("--reverse", "--reverse", dest="reverse", default=0, type=int)
parser.add_argument("--ensemble_mode", "--ensemble_mode", dest="ensemble_mode", default='dot_product', type=str)
parser.add_argument("--num_local", "--num_local", help="this is num_local (used model number for inference). if the value is set, then the number of models used for inference is fixed", dest="num_local", type=int, default=6)
parser.add_argument("--use_advanced_extractor", "--use_advanced_extractor", dest="use_advanced_extractor", action='store_true')
parser.add_argument("--joint_optimizer", "--joint_optimizer", dest="joint_optimizer", action='store_true')

# training parameters
parser.add_argument("--ensemble_lr", "--ensemble_lr", help="this is ensemble_lr", dest="ensemble_lr", type=float, default=0.0005)
parser.add_argument("--clf_lr", "--clf_lr", help="this is clf_lr", dest="clf_lr", type=float, default=1.0)
parser.add_argument("--batch_size", "--batch_size", help="this is batch_size", dest="batch_size", type=int, default=128)
parser.add_argument("--warmup_epoch", "--warmup_epoch", dest="warmup_epoch", default=1, type=int)
parser.add_argument("--step_size", "--step_size", dest="step_size", default=100, type=int)
parser.add_argument("--gamma", "--gamma", help="this is gamma", dest="gamma", type=float, default=0.95)
parser.add_argument("--epsilon", "--epsilon", help="this is epsilon", dest="epsilon", type=float, default=1.1)
parser.add_argument("--weight_decay", "--weight_decay", help="this is weight_decay", dest="weight_decay", type=float, default=1e-8)
parser.add_argument("--gumbel_tau", "--gumbel_tau", help="this is gumbel_tau", dest="gumbel_tau", type=float, default=1.0)
parser.add_argument("--num_epoch", "--num_epoch", help="this is num_epoch (if not use domainbed-default epoch number then set)", dest="num_epoch", type=int, default=0)
parser.add_argument("--rough_rank_epoch", "--rough_rank_epoch", dest="rough_rank_epoch", default=1, type=int)
parser.add_argument("--rough_rank_num", "--rough_rank_num", dest="rough_rank_num", default=-1, type=int)
parser.add_argument("--domainnet_split_point", "--domainnet_split_point", dest="domainnet_split_point", default=20, type=int)

# loss weights
parser.add_argument("--ensemble_weight", "--ensemble_weight", help="this is ensemble_weight", dest="ensemble_weight", type=float, default=1.0)
parser.add_argument("--child_weight", "--child_weight", help="this is child_weight", dest="child_weight", type=float, default=1.0)
parser.add_argument("--confidence_weight", "--confidence_weight", help="this is confidence_weight", dest="confidence_weight", type=float, default=1.0)

args = parser.parse_args()

# the lr of clf is a scale (args.clf_lr) of the lr of the ensemble network
# be attention that current the ensemble network is a GNN model, so the tuning of lr will change a large
args.clf_lr = args.clf_lr*args.ensemble_lr

#print the command-line arguments obtained from the user
print('args:',args)

# initialize dictionary to store results
res = {}

# set device
cuda_available = torch.cuda.is_available() # check if CUDA is available for GPU computation
if cuda_available: # if GPU available
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(args.random_seed) # set random seed for GPU
    print('Using GPU.')
    device = torch.device("cuda")
else: # if CPU available
    print('Using CPU.')
    device = torch.device("cpu")
args.device = device # set device for args

# set random seed for reproducibility
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
print('Successfully imported all packages and configured random seed to %d!'%args.random_seed)

# set directories for logs and models
save_dir = './logs/'
model_save_dir = './models/' 

# if directories don't exist, create them
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# redirect standard output to logger
sys.stdout = Logger(save_dir + 'running.log')
args.writer = SummaryWriter(save_dir)

# load pretrained models
pretrained_models=[]
with open(args.pretrain_model_list,'r') as f:
    for line in f:
        pretrained_models.append(line.strip('\n'))

# if performing inference on single model, only keep that model
if args.idx_for_infer_save != -1:
    pretrained_models = pretrained_models[args.idx_for_infer_save:args.idx_for_infer_save+1]
args.model_num = len(pretrained_models)
args.pretrained_models = pretrained_models

# if working with domainbed dataset, set data directory
if args.dataset == 'domainbed':
    data_dir = '/home/private_user_1/v-liziyue/ziyue/domainbed_inference_saved_' + args.domainbed_dataset + '/'

# initialize model pool and ensemble network
model_pool = Model_pool(model_name = pretrained_models, model_num = args.model_num, model_idx = args.model_idx, dataset = args.dataset,sub_dataset = args.domainbed_dataset, data_dir = data_dir, args = args)
ensemble_net = Ensemble_of_ensemble(args)

# if manually set number of epochs, then change default one
ensemble_net.num_epoch = args.num_epoch if args.num_epoch != 0 else ensemble_net.num_epoch
args.num_epoch = ensemble_net.num_epoch 

# if only saving inference results, run inference saver function
if args.save_inference_only:
    inference_saver(args, ensemble_net, model_pool)
else: # else train model
    best_val_acc = 0
    best_val_epoch = 0
    valbest_test_acc = 0
    early_stop_count = 0
    
    training_start_time = time.time()
    sum_bp_time = 0

    for epoch in range(ensemble_net.num_epoch):
        print('------------------------------------------------------------------------')
        print('epoch ', epoch, ' training')

        # train model and log backpropagation time
        _, bp_time = train_ensemble(epoch, \
            ensemble_net,\
            ensemble_net.train_loader,\
            split = 'train',\
            model_pool = model_pool, args = args)

        sum_bp_time += bp_time
        print('------------------------------------------------------------------------')
        print('epoch ', epoch, ' validation')
        # validate model and get accuracy and topk indices
        val_acc, topk_indices = validate_ensemble(epoch, ensemble_net, 
            ensemble_net.val_loader, split = 'validation', model_pool = model_pool, args = args)

        # if current epoch has best validation accuracy so far, update best validation accuracy and test accuracy
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            early_stop_count = 0
            print('best val acc')
            best_val_epoch = epoch
            
            # save best performing model
            if args.idx_for_infer_save == -1:
                torch.save(ensemble_net.state_dict(), f"{model_save_dir}/{args.random_seed}_{args.domainbed_test_env}_best_ensemble_net.pth")
            else:
                torch.save(ensemble_net.state_dict(), f"{model_save_dir}/{args.idx_for_infer_save}_{args.domainbed_dataset}_{args.domainbed_test_env}_{args.random_seed}_best_ensemble_net.pth")

            print('------------------------------------------------------------------------')
            print('epoch ', epoch, ' testing')
            # test model and get accuracy
            test_acc, _ = validate_ensemble(epoch, ensemble_net,\
                ensemble_net.test_loader, split = 'evaluation', model_pool = model_pool, args = args)
        else:
            early_stop_count += 1

        # reselect top-k models every rough_rank_epoch epochs
        if epoch == args.rough_rank_epoch - 1:
            reselected_models = [pretrained_models[i] for i in topk_indices]
            model_pool.reinit(reselected_models)
            args.model_num = len(reselected_models)
            ensemble_net.reinit(reselected_models, topk_indices, args)
            res['topk_indices'] = str(topk_indices.tolist())

        # if current epoch is best validation epoch, update test accuracy
        if epoch == best_val_epoch:
            valbest_test_acc = test_acc

        # if current epoch is final epoch, save model
        if epoch == args.num_epoch - 1:
            if args.idx_for_infer_save == -1:
                torch.save(ensemble_net.state_dict(), f"{model_save_dir}/{args.random_seed}_{args.domainbed_test_env}_final_ensemble_net.pth")
            else:
                torch.save(ensemble_net.state_dict(), f"{model_save_dir}/{args.idx_for_infer_save}_{args.domainbed_dataset}_{args.domainbed_test_env}_{args.random_seed}_final_ensemble_net.pth")

        # implement early stopping for non-DomainNet datasets
        if args.domainbed_dataset != 'DomainNet':
            if early_stop_count == 5:
                print('early stop')
                break
        else: # implement early stopping for DomainNet dataset
            if early_stop_count == 3:
                print('DomainNet early stop')
                break            

    print('the overall time cost: ', time.time() - training_start_time)
    print('the overall bp time: ', sum_bp_time)
    print('valbest_test_acc: ', valbest_test_acc)

    # record results in dictionary
    res['seed'] = args.random_seed
    res['dataset'] = args.domainbed_dataset

    res['super_mixed'] = args.super_mixed
    res['average_clf'] = args.average_clf
    res['average_ensemble'] = args.average_ensemble
    res['mixed_training'] = args.mixed_training

    res['domainbed_test_env'] = args.domainbed_test_env
    res['valbest_test_acc'] = valbest_test_acc
    res['sum_bp_time'] = sum_bp_time
    res['num_params'] = ensemble_net.num_params

    # save results in json format
    if args.idx_for_infer_save == -1:
        with open(f"{save_dir}/{args.random_seed}_{args.domainbed_test_env}_res.json", "w") as f:
            json.dump(res, f, indent=4, sort_keys=True)
    else:
        with open(f"{save_dir}/{args.idx_for_infer_save}_{args.domainbed_dataset}_{args.domainbed_test_env}_{args.random_seed}_res.json", "w") as f:
            json.dump(res, f, indent=4, sort_keys=True)