# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import logging

import torch
import torchvision.transforms as transforms
from torch import autograd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def compose_json(self):
        best_res = {}
        for meter in self.meters:
            best_res[meter.name] = meter.avg
        return best_res

    def display_avg(self):
        entries = [self.prefix ]
        entries += [f"{meter.name}:{meter.avg:6.3f}" for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    
def get_network(args, num_classes):
    """ return given network
    """

    if args.net == 'resnet18':
        from models.resnet import ResNet, BasicBlock
        return ResNet, {"block": BasicBlock, "num_blocks": [2, 2, 2, 2], 'num_classes':num_classes}
    elif args.net == 'resnet32':
        from models.resnet32 import ResNet, BasicBlock
        return ResNet, {"block": BasicBlock, "num_blocks": [5, 5, 5], 'num_classes':num_classes}
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

 
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name = None):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset_name == 'cifar100':
        from dataset import CIFAR100_idx as dataset_idx
    elif dataset_name == 'cifar10':
        from dataset import CIFAR10_idx as dataset_idx
    else:
        print('the dataset name you have entered is not supported yet')
        sys.exit()
        
    training_set = dataset_idx(root='./data', train=True, download=True, transform=transform_train)
    _training_loader = DataLoader(
        training_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return _training_loader, None


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name = None):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_data_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    if dataset_name == 'cifar100':
        from dataset import CIFAR100_idx as dataset_idx
    elif dataset_name == 'cifar10':
        from dataset import CIFAR10_idx as dataset_idx
    else:
        print('the dataset name you have entered is not supported yet')
        sys.exit()
        
    test_set = dataset_idx(root='./data', train=False, download=True, transform=transform_test)

    cifar100_test_loader = DataLoader(
        test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    

def get_logger(file_path):
    """ Make python logger """
    logger = logging.getLogger('USNet')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class ThresholdBinarizer(autograd.Function):
    """
    Threshold binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j} > \tau`
    where `\tau` is a real value threshold.

    Implementation is inspired from:
        https://github.com/arunmallya/piggyback
        Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
        Arun Mallya, Dillon Davis, Svetlana Lazebnik
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float, sigmoid: bool, min_elements: float=0.0):
        """
        We limit by default the pruning so that at least 0.5% (half a percent) of the weights are remaining (min_elements)
        If you set min_elements to zero, no minimal number of elements will be enforced.
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The threshold value (in R).
            sigmoid (`bool`)
                If set to ``True``, we apply the sigmoid function to the `inputs` matrix before comparing to `threshold`.
                In this case, `threshold` should be a value between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        nb_elems = inputs.numel()
        if min_elements != 0:
            nb_min = int(min_elements * nb_elems) + 1
        else:
            nb_min = 0
        if sigmoid:
            mask = (torch.sigmoid(inputs) > threshold).type(inputs.type())
        else:
            mask = (inputs > threshold).type(inputs.type())
        if mask.sum() < nb_min:
            k_threshold = inputs.flatten().kthvalue(max(nb_elems - nb_min, 1)).values
            mask = (inputs > k_threshold).type(inputs.type())
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None, None


def cost_calculation(Exit_rate_l, Flag, n_estimators):
    """ Calculate the cost of inference based on exit """
    
    cost = 0
    exit_rate_all = 0
    if Flag:
        for i in range(len(Exit_rate_l)-1):
            exit_rate_all += Exit_rate_l[i] * 0.01
            cost += Exit_rate_l[i] * 0.01 * (i + 1)
        cost += (1 - exit_rate_all) * n_estimators
    else:
        for i in range(len(Exit_rate_l)):
            exit_rate_all += Exit_rate_l[i] * 0.01
            cost += Exit_rate_l[i] * 0.01 * (i + 1)
        cost += (1 - exit_rate_all) * n_estimators
    return cost