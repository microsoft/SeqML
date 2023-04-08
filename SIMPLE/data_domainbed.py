# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

# Import datasets and hyperparameter registry from domainbed package
from miscellaneous.domainbed import datasets
from miscellaneous.domainbed import hparams_registry


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    """Define a class to create an infinite dataloader"""
    def __init__(self, dataset, weights, batch_size, num_workers, sampler, drop_last = True):
        super().__init__()
        
        # If no weights are given, set them to one
        if weights == None:
            weights = torch.ones(len(dataset))

        # Create a batch sampler from the given sampler
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)
        
        # Create an infinite iterator from the given dataset and batch sampler
        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        while True:
            # Yield the next batch from the infinite iterator
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length


# Define a function to create data loaders for domain generalization tasks
def domainbed_dataloader(dataset, data_dir, test_envs, batch_size, domain_random = True, work_num = 4):

    # Get default hyperparameters for the given dataset
    hparams = hparams_registry.default_hparams('IGA', dataset)
    # Create the dataset object
    dataset = vars(datasets)[dataset](data_dir,
        test_envs, hparams)

    source_domain = []
    target_domain = []
    
    # Get the class names from the dataset
    classes = dataset[0].classes
    
    for env_i, env in enumerate(dataset):
        # Sort the environments into source and target domains
        if env_i in test_envs:
            target_domain.append(env)
        else:
            source_domain.append(env)

    if domain_random:
        # If domain_random is True, create random training and validation samplers
        # that sample from 80% of the training data, and a separate dataloader
        # for the test set
        
        sum_train_sample = np.sum([len(env) for i, env in enumerate(source_domain)])
        
        # Create samplers for the training and validation sets (useful for splits with more than one training domain)
        train_sampler = RandomSampler(range(0,int(sum_train_sample*0.8)))
        val_sampler = RandomSampler(range(int(sum_train_sample*0.8), sum_train_sample))

        # Concatenate all the training domains into one dataset
        if len(source_domain) == 1:
            source_domains = source_domain[0]
        elif len(source_domain) == 3:
            source_domains = source_domain[0] + source_domain[1] + source_domain[2]
        elif len(source_domain) == 5:
            source_domains = source_domain[0] + source_domain[1] + source_domain[2] + source_domain[3] + source_domain[4]

        # Set batch size to the minimum of the given batch size and the number of samples in the training dataset
        if len(source_domains) < batch_size:
            batch_size = len(source_domains)

        # Create dataloaders for the training, validation, and test sets using the SoftwarePipeLine class
        train_loaders = SoftwarePipeLine(DataLoader(
                source_domains,
                batch_size=batch_size, sampler = train_sampler, drop_last = True,
                num_workers=work_num, pin_memory=True), env = True)

        valid_loaders = SoftwarePipeLine(DataLoader(
                source_domains,
                batch_size=batch_size, sampler = val_sampler, 
                num_workers=work_num, pin_memory=True), env = True)

        test_loaders = SoftwarePipeLine(DataLoader(
                target_domain[0],
                batch_size=batch_size, shuffle=False,
                num_workers=work_num, pin_memory=True), env = True)

        full_loaders = SoftwarePipeLine(DataLoader(
                dataset,
                batch_size=batch_size, shuffle=False,
                num_workers=work_num, pin_memory=True), env = True)
    else:
        # If domain_random is False, create infinite dataloaders for each training domain,
        # with separate training and validation samplers
        
        train_samplers = [RandomSampler(range(0,int(len(env)*0.8))) for i, env in enumerate(source_domain)]
        val_samplers = [RandomSampler(range(int(len(env)*0.8), len(env))) for i, env in enumerate(source_domain)]

        train_loaders = [
            SoftwarePipeLine(InfiniteDataLoader(
                env,
                weights = None,
                batch_size=batch_size,
                num_workers=work_num, sampler = train_samplers[i], drop_last = True), env = True) for i, env in enumerate(source_domain)
        ]

        valid_loaders = [
            SoftwarePipeLine(DataLoader(
                env,
                batch_size=batch_size, sampler = val_samplers[i],
                num_workers=work_num, pin_memory=True, drop_last = True), env = True) for i, env in enumerate(source_domain)
        ]

        test_loaders = [
            SoftwarePipeLine(DataLoader(
                env,
                batch_size=batch_size, shuffle=False, 
                num_workers=work_num, pin_memory=True), env = True) for i, env in enumerate(target_domain)
        ]

        full_loaders = [
            SoftwarePipeLine(DataLoader(
                env,
                batch_size=batch_size, shuffle=False,
                num_workers=work_num, pin_memory=True), env = True) for i, env in enumerate(dataset)
        ]

    # Return the dataloaders, with a list of class names
    return train_loaders, valid_loaders, test_loaders, full_loaders, classes


# Define a wrapper function that adds sample indices to dataset items
def dataset_with_indices(cls, env_i = None):

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        if env_i == None:
            return data, target, index
        else:
            return data, target, index, env_i

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })
    

# Define a class to create a software pipeline for use with the dataloader 
class SoftwarePipeLine(object):
    
    def __init__(self, dataloader, env = False):
        self.dataloader = dataloader
        self.stream = None
        self.env = env
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        if self.stream is None:
            self.stream = torch.cuda.Stream()
            
        first = True
        # Iterate over the dataloader and yield the transformed input, target,
        # index, and environment tensors within the software pipeline
        
        if self.env == False:
            for next_input, next_target, next_idx in self.dataloader:
                with torch.cuda.stream(self.stream):
                    next_input = next_input.cuda(non_blocking=True)
                    next_target = next_target.cuda(non_blocking=True)
                    next_idx = next_idx.cuda(non_blocking=True)
                if not first:
                    yield input, target, index
                else:
                    first = False
                torch.cuda.current_stream().wait_stream(self.stream)
                input = next_input
                target = next_target
                index = next_idx
            yield input, target, index
        else:
            for next_input, next_target, next_idx, next_env in self.dataloader:
                with torch.cuda.stream(self.stream):
                    next_input = next_input.cuda(non_blocking=True)
                    next_target = next_target.cuda(non_blocking=True)
                    next_idx = next_idx.cuda(non_blocking=True)
                    next_env = next_env.cuda(non_blocking=True)
                if not first:
                    yield input, target, index, env
                else:
                    first = False
                torch.cuda.current_stream().wait_stream(self.stream)
                input = next_input
                target = next_target
                index = next_idx
                env = next_env
            yield input, target, index, env