# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import torch
import numpy as np
import random
import argparse

from utils import get_training_dataloader, get_test_dataloader, get_logger, get_network
from ensemble import EnsembleClassifier, Selector
from torchensemble.conf import settings
from torchensemble.utils.logging import set_logger


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # settings for network architecture, dataset, and ensemble size to exp
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-dataset', type=str, default='cifar100', help='dataset type')
    parser.add_argument('-n_estimators', type=int, default=3, help='the number of estimators (default: 3)')
    
    # settings for training
    parser.add_argument('-work_dir', type=str, default='./work_dir/', help='dir name')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-print_freq', type=int, default=100, help='frequency of printing training states')
    parser.add_argument('-seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('-n_jobs', type=int, default=2, metavar='S', help='n_jobs')

    # settings for losses
    parser.add_argument('-use_ensemble_loss', default=0.0, type=float, help='weight of ensemble_loss')
    parser.add_argument('-use_cost_loss', default=1.0, type=float, help='weight of cost_loss')
    parser.add_argument('-use_ranking_loss_m', default=0.0, type=float, help='weight of ranking_loss for models')
    parser.add_argument('-use_ranking_loss_s', default=0.0, type=float, help='weight of ranking_loss for the selector')
    parser.add_argument('-ranking_loss_margin_m', default=1.0, type=float, help="margin of ranking loss for models")
    parser.add_argument('-ranking_loss_margin_s', default=1.0, type=float, help="margin of ranking loss for the selector")
    parser.add_argument('-alpha', default=0.1, type=float)
    parser.add_argument('-aux_dis_lambda', type=float, default=50.0, help='aux_dis_lambda loss rate')
    
    # settings for optimizers and schedulers
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-selector-lr', type=float, default=0.001, help='learning rate of the selector')
    parser.add_argument('-selector-weight-decay', type=float, default=1e-8, help='weight decay of the optimizer for the selector')
    parser.add_argument('-step_size', type=int, default=20, help='step size for decaying learning rate for the selector')
    parser.add_argument('-gamma', type=float, default=0.95, help='gamma of the optimizer for the selector')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'cifar10':
        num_classes = 10
    else:
        print('the dataset name you have entered is not supported yet')
        sys.exit()
        
        
    # set up base models
    estimator, estimator_args = get_network(args, num_classes)

    # set up the selector
    selector = Selector(num_class = num_classes, args = args)

    # set up ensemble model
    loss_function = torch.nn.CrossEntropyLoss(reduce=False)
    net = EnsembleClassifier(
        estimator=estimator, n_estimators=args.n_estimators, estimator_args=estimator_args, cuda=True, n_jobs=args.n_jobs, args=args, log_dir=args.work_dir, selector = selector
    )
    net.set_optimizer('SGD', lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    net.set_scheduler('MultiStepLR', milestones=settings.MILESTONES, gamma=0.2)
    net.set_criterion(loss_function)
    net.set_selector_optimizer('SGD', lr=args.selector_lr, momentum=0.9, weight_decay=args.selector_weight_decay, nesterov=True)
    
    # set up logger  
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)
    logger = get_logger(os.path.join(args.work_dir, 'train.log'))
    logger.info(args)
    logger.info(net)
    settings.LOG_DIR = os.path.join(args.work_dir, 'pb')
    checkpoint_path = os.path.join(args.work_dir, 'ckpt')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    logger = set_logger(
        use_tb_logger=True, log_path=args.work_dir
    )

    training_loader, validation_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        dataset_name = args.dataset
    )

    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        dataset_name = args.dataset
    )

    # training
    net.fit(
        training_loader,
        epochs=settings.EPOCH,
        log_interval=args.print_freq,
        test_loader=test_loader,
        valid_loader=validation_loader,
        save_model=True,
        save_dir=checkpoint_path,
        )

    # evaluation
    testing_acc = net.evaluate(test_loader, save_dir=checkpoint_path, args = args, selectors=net.selectors)
    for ind in range(50):
        margin_to_test = ind * 2 / 100.0
        _ = net.evaluate(test_loader, save_dir=checkpoint_path, args = args, selectors=net.selectors, margin = margin_to_test)
