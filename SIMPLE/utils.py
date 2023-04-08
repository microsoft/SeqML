# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
from torch.optim.lr_scheduler import _LRScheduler


# Function for Computing the Precision
def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    
    # Set the maximum value of k
    maxk = max(topk)
    # Determine batch size
    batch_size = target.size(0)

    # Get predictions with top values
    _, pred = output.topk(maxk, 1, True, True)
    # Reshape the output
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        # Calculate the number of correct predictions
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        # Calculate the accuracy rate
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Function for Computing and Storing the Average Value
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


# Function for Storing the Progress of the Model
class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger = None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        # Assign formatted prefix statement to the entries list
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]

    def compose_json(self):
        # Empty Dictionary
        best_res = {}
        for meter in self.meters:
            # Update the Dictionary
            best_res[meter.name] = meter.avg
        return best_res

    def display_avg(self):
        entries = [self.prefix]
        # Assign formatted prefix statement to the entries list
        entries += [f"{meter.name}:{meter.avg:6.3f}" for meter in self.meters]

    def _get_batch_fmtstr(self, num_batches):
        # Get the length of string format
        num_digits = len(str(num_batches // 1))
        # Assign formatted value to the entries list
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# Function for the Warmup Learning Rate
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        # Set Total Iterations and Variables from _LRScheduler
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        
        # Calculate Learning Rate
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


# Function for the Logger
class Logger(object):
    def __init__(self, filename="Default.log"):
        # Set a Terminal Object
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        # Write Message to Terminal
        self.terminal.write(message)
        # Write Message to Log
        self.log.write(message)
 
    def flush(self):
        pass


# Function for Walking through All the Files
def walkFile(file):
    count = 0
    for root, dirs, files in os.walk(file):
        for f in files:
            count += 1
            # Determine file size
            file_size = os.path.getsize(os.path.join(root, f)) 
            if file_size == 0:
                # Return a value of 0 if model encounters a zero-length file
                return 0
    return count


# Function for Checking the File Number
def check_file_number(dir_to_check, domainbed_dataset):
    dic = {'PACS':41, 'VLCS': 44, 'OfficeHome': 64, 'TerraIncognita': 97, 'DomainNet': 2295}
    # Define a variable for the file count
    num = walkFile(dir_to_check)
    # Compare the number of files with the expected value
    return (num == dic[domainbed_dataset])