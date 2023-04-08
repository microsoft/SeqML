# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

import os
import copy
import time
import timm
import types
import psutil
import numpy as np
import pretrainedmodels
from itertools import chain
import miscellaneous.clip as clip

from utils import WarmUpLR
import miscellaneous.model.models_vit as models_vit
from data_domainbed import domainbed_dataloader



# dataset_inference_information contains information about the datasets and their corresponding models,
# including whether to preload the models or not, the number of domains, and the number of classes
dataset_inference_information = {
    'domainbed': {'PACS': {'preload': True, 'domain_num': 4, 'class_type': 7}, 'VLCS': {'preload': True, 'domain_num': 4, 'class_type': 5},
        'OfficeHome': {'preload': True, 'domain_num': 4, 'class_type': 65},
        'TerraIncognita': {'preload': True, 'domain_num': 4, 'class_type': 10}, 'DomainNet': {'preload': True, 'domain_num': 6, 'class_type': 345}},
    'flops_test': {'preload': False}
}

class Model_pool(torch.nn.Module):

    def __init__(self, model_name, model_num, model_idx = None, dataset = 'domainbed', sub_dataset = None, data_dir = None, args = None):

        super(Model_pool, self).__init__()
        
        self.model_name = model_name
        self.model_num = model_num
        self.dataset = dataset
        self.sub_dataset = sub_dataset
        self.model_idx = model_idx
        self.data_dir = data_dir
        self.args = args

        print('sub_dataset:', sub_dataset)

        # Check if models should be preloaded
        if args.save_inference_only == False:
            preload_flag = dataset_inference_information[dataset]['preload'] if sub_dataset == None else dataset_inference_information[dataset][sub_dataset]['preload']
        else:
            preload_flag = False

        # If models should be preloaded, set self.models to None and self.preload_dic to an empty dictionary
        if preload_flag:
            self.models = None
            self.preload_dic = {}
        # If models should not be preloaded, create a module list for self.models and a module list for self.process, then call load_all_model()
        else:
            self.models = nn.ModuleList()
            self.process = nn.ModuleList()
            self.load_all_model()

        print('self.model_name:', self.model_name)
        # If args.model_idx is not None, set self.model_name to a slice of the original self.model_name based on the value of args.model_idx
        if args.model_idx != None:
            self.model_name = self.model_name[args.model_idx:args.model_idx+1]

        self.clear_model_name = []

        # For each model in self.model_name, replace '/' with '-' and append it to self.clear_model_name
        for self.model_idx in range(self.model_num):
            if self.model_name[self.model_idx][:5] != '(MAE)':
                self.clear_model_name.append(self.model_name[self.model_idx].replace('/','-'))
            else:
                self.clear_model_name.append(self.model_name[self.model_idx].replace('/','-').split(':')[0])

    # Reinitialize the model with a new model_name
    def reinit(self, model_name):
        self.model_name = model_name
        self.model_num = len(model_name)

        self.clear_model_name = []

        # For each model in self.model_name, replace '/' with '-' and append it to self.clear_model_name
        for self.model_idx in range(self.model_num):
            if self.model_name[self.model_idx][:5] != '(MAE)':
                self.clear_model_name.append(self.model_name[self.model_idx].replace('/','-'))
            else:
                self.clear_model_name.append(self.model_name[self.model_idx].replace('/','-').split(':')[0])

    # Load a single model based on its prefix and model_name
    def load_single_model(self):

        model_info = self.model_name[self.model_idx][1:].split(')')
        prefix = model_info[0]
        model_name = model_info[1]

        print('prefix:', prefix, ' model_name:', model_name)
        
        # If prefix is 'clip', load the clip model
        if prefix == 'clip':
            model, _ = clip.load(model_name, self.args.device)
        # If prefix is 'ssl', load the semi-supervised ImageNet1K model
        elif prefix == 'ssl':
            model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        # If prefix is 'swag', load the SWAG model
        elif prefix == 'swag':
            model = torch.hub.load("facebookresearch/swag", model=model_name)
        # If prefix is 'pretrainedmodels', load the pretrained models
        elif prefix == 'pretrainedmodels':
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        # If prefix is 'timm', load the timm model
        elif prefix == 'timm':
            model = timm.create_model(model_name, pretrained=True)
        # If prefix is 'MAE', load the vit model with the specified aux_info path for the checkpoint
        elif prefix == 'MAE':
            model_name_split = model_name.split(':')
            model_name = model_name_split[0]
            aux_info = model_name_split[1]
            model = models_vit.__dict__[model_name](
                num_classes=1000,
                drop_path_rate=0.1,
                global_pool=True,
            )

            # Load the model state dict from the specified path
            checkpoint = torch.load('/home/private_user_1/v-liziyue/pytorch_pretrained_models/hub/checkpoints/' + \
                aux_info, map_location='cpu')

            model.load_state_dict(checkpoint['model'])

        return model
        

    # Load all models in self.model_name
    def load_all_model(self):

        # If self.model_idx is not None, load only the specified model
        if self.model_idx != None:
            model = self.load_single_model(self.model_idx)
            self.models.append(model)
        # If self.model_idx is None, load all models in self.model_name
        else:
            for self.model_idx in range(self.model_num):
                model = self.load_single_model()
                self.models.append(model)

        # Freeze the models
        freeze(self.models)

    # Inference using the models in self.models
    def inference_by_models(self, images):

        y_hats = []
        for i in range(len(self.model_name)):
            with torch.no_grad():
                self.models[i] = self.models[i].to(images.device)
                self.models[i].eval()
                y_hat = self.models[i](images) # (B, 1000), list
                self.models[i].cpu()
            y_hats.append(y_hat)
            torch.cuda.empty_cache() 
        return y_hat

    # Inference using preloaded models in self.preload_dic
    def inference_by_preload(self, images, idxs, domain_idxs = None, split = None, epoch = -1):
        y_hats = []
        length = 256

        # For each model in self.clear_model_name, load the model's predictions for the specified idxs into y_hat
        for model_name in self.clear_model_name:
            y_hat = []
            for sample_idx, idx in enumerate(idxs):
                if domain_idxs != None:
                    domain_idx = domain_idxs[sample_idx].item()
                else:
                    domain_idx = None
                data_idx = idx//length
                outside = idx%length
                if model_name not in self.preload_dic:
                    self.preload_dic[model_name] = {}
                data_path = self.dataset if domain_idx == None else self.sub_dataset + '_' + str(domain_idx)

                if domain_idx != None:
                    if domain_idx not in self.preload_dic[model_name]:
                        self.preload_dic[model_name][domain_idx] = {}
                    
                    if data_idx.item() not in self.preload_dic[model_name][domain_idx]:
                        self.preload_dic[model_name][domain_idx][data_idx.item()] = torch.load(self.data_dir + model_name + '/' + data_path + '/' + str(data_idx.item()) + '.pth')
                    dic = self.preload_dic[model_name][domain_idx][data_idx.item()]
                else:
                    if data_idx.item() not in self.preload_dic[model_name]:
                        self.preload_dic[model_name][data_idx.item()] = torch.load(self.data_dir + model_name + '/' + data_path + '/' + str(data_idx.item()) + '.pth')
                    dic = self.preload_dic[model_name][data_idx.item()]
                    
                y_hat.append(dic[outside.item()].unsqueeze(0))
            
            y_hat = torch.cat(y_hat, dim = 0) # (B, 1000)
            y_hats.append(y_hat)
            torch.cuda.empty_cache() 

        # Clear self.preload_dic if CPU or memory utilization is too high
        if psutil.cpu_percent() > 95.0 or psutil.virtual_memory().percent > 95.0:
            self.preload_dic = {}

        return y_hats

    # Forward function of the model
    def forward(self, images, idxs = None, domain_idx = None, split = None, epoch = -1):
        return self.inference_by_preload(images, idxs, domain_idx, split, epoch = epoch)

# Deepcopy the network and create a Moving_average_net module with it
def __deepcopy__(self, memo={}):
    cls = self.__class__
    copyobj = cls.__new__(cls)
    memo[id(self)] = copyobj
    for attr, value in self.__dict__.items():
        try:
            setattr(copyobj, attr, copy.deepcopy(value, memo))
        except Exception:
            pass
    return copyobj

# Define the Moving_average_net module
class Moving_average_net(torch.nn.Module):
    def __init__(self, network, sma_start_iter):

        super(Moving_average_net, self).__init__()

        self.network = network
        network.__deepcopy__ = types.MethodType(__deepcopy__, network)
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = sma_start_iter
        self.current_iter = 0
        self.average_count = 0
    
    # Update the stored moving average of the network parameters
    def update_sma(self):
        self.current_iter += 1
        
        # If current iter is greater than or equal to sma_start_iter, compute new average and update self.average_count
        if self.current_iter >= self.sma_start_iter:
            self.average_count += 1
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = (param_k.data*self.average_count + param_q.data)/ (1.0 + self.average_count)
        else:
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = param_q.data

        
class Ensemble_of_ensemble(torch.nn.Module):
    
    # Initialize the class and set initial values.
    def __init__(self, args):
        
        super(Ensemble_of_ensemble, self).__init__()
        self.dataset = args.dataset
        self.args = args
        self.device = args.device
        
        # Load the data
        self.load_data(args)

        # If the dataset is domainbed, get class type and domain number from the dataset_inference_information.
        if args.dataset == 'domainbed':
            class_type = dataset_inference_information['domainbed'][self.args.domainbed_dataset]['class_type']
            self.class_type = class_type
            domain_num = dataset_inference_information['domainbed'][self.args.domainbed_dataset]['domain_num']
            args.domain_num = domain_num
            self.ensemble_net = Ensemble_net(args)
            
            if args.domainbed_dataset != 'DomainNet':
                self.task_head_list = [1000, 21843, 21841, 768, 1024, 1280, 3024, 3712, 7392, 11221]
                self.task_specific_head = nn.ModuleList()
                self.task_specific_head.append(task_head(1000,class_type))
                
                # If it is not pure_imagenet_pretrained, append numbers in the list to task_specific_head.
                if not args.pure_imagenet_pretrained:
                    self.task_specific_head.append(task_head(21843,class_type))
                    self.task_specific_head.append(task_head(21841,class_type))
                    self.task_specific_head.append(task_head(768,class_type)) # vit_b16
                    self.task_specific_head.append(task_head(1024,class_type)) # vit_l16
                    self.task_specific_head.append(task_head(1280,class_type)) # vit_h14
                    self.task_specific_head.append(task_head(3024,class_type)) # regnety_16gf
                    self.task_specific_head.append(task_head(3712,class_type)) # regnety_32gf
                    self.task_specific_head.append(task_head(7392,class_type)) # regnety_128gf
                    self.task_specific_head.append(task_head(11221,class_type)) # regnety_128gf

            # If domainbed_dataset is DomainNet, set task_head_list to a specific list of numbers and append each to task_specific_head.
            else:
                self.task_head_list = [1000, 21841, 768, 1024, 1280, 3024, 3712]
                self.task_specific_head = nn.ModuleList()
                self.task_specific_head.append(task_head(1000,class_type))
                
                # If it is not pure_imagenet_pretrained, append numbers in the list to task_specific_head.
                if not args.pure_imagenet_pretrained:
                    self.task_specific_head.append(task_head(21841,class_type))
                    self.task_specific_head.append(task_head(768,class_type)) # vit_b16
                    self.task_specific_head.append(task_head(1024,class_type)) # vit_l16
                    self.task_specific_head.append(task_head(1280,class_type)) # vit_h14
                    self.task_specific_head.append(task_head(3024,class_type)) # regnety_16gf
                    self.task_specific_head.append(task_head(3712,class_type)) # regnety_32gf

            
            # Set the ensemble_average and classifier_average.
            self.ensemble_average = Moving_average_net(self.ensemble_net, args.start_iteration_ensemble)
            self.classifier_average = Moving_average_net(self.task_specific_head, args.start_iteration_clf)

            # If joint_optimizer is true, set the optimizer using Adam and use WarmUpLR and StepLR.
            if args.joint_optimizer:
                self.optimizer_ensemble = torch.optim.Adam(filter(lambda x: x.requires_grad, 
                    chain(self.ensemble_net.parameters(), self.task_specific_head.parameters())), lr=args.ensemble_lr,\
                    eps=1e-8, weight_decay=args.weight_decay)
                self.warmup_scheduler = WarmUpLR(self.optimizer_ensemble, self.steps_per_epoch*args.warmup_epoch)
                self.scheduler_ensemble = torch.optim.lr_scheduler.StepLR(self.optimizer_ensemble, step_size=args.step_size, gamma=args.gamma)

            # Otherwise, set the optimizer for ensemble_net and task_specific_head separately.
            else:
                self.optimizer_ensemble = torch.optim.Adam(filter(lambda x: x.requires_grad, 
                    chain(self.ensemble_net.parameters())), lr=args.ensemble_lr,\
                    eps=1e-8, weight_decay=args.weight_decay)
                self.warmup_scheduler = WarmUpLR(self.optimizer_ensemble, self.steps_per_epoch*args.warmup_epoch)
                self.scheduler_ensemble = torch.optim.lr_scheduler.StepLR(self.optimizer_ensemble, step_size=args.step_size, gamma=args.gamma)


                self.optimizer_clf = torch.optim.Adam(filter(lambda x: x.requires_grad, 
                    chain(self.task_specific_head.parameters())), lr=args.clf_lr,\
                    eps=1e-8, weight_decay=args.weight_decay)
                self.clf_warmup_scheduler = WarmUpLR(self.optimizer_clf, self.steps_per_epoch*args.warmup_epoch)
                self.scheduler_clf = torch.optim.lr_scheduler.StepLR(self.optimizer_clf, step_size=args.step_size, gamma=args.gamma)

            # Calculate the total number of parameters.
            total = sum([param.nelement() for param in self.ensemble_net.parameters()] + [param.nelement() for param in self.task_specific_head.parameters()]) - sum([param.nelement() for param in self.ensemble_net.feature_extractor.parameters()])
            print("Number of the whole parameter: %.2fM" % (total/1e6))
            self.num_params = total/1e6
        
        # If the dataset is flops_test, set the num_epoch to 0.
        elif args.dataset == 'flops_test':
            self.num_epoch = 0
    
    # Reinitialize using reselected_models, topk_indices, and args.
    def reinit(self, reselected_models, topk_indices, args):
        self.ensemble_net.model_emb = torch.nn.Parameter(self.ensemble_net.model_emb[topk_indices,:], requires_grad=True)
        self.ensemble_net.w3 = torch.nn.Linear(len(reselected_models), len(reselected_models))
        
        self.ensemble_net.model_num = len(reselected_models)
        self.ensemble_average = Moving_average_net(self.ensemble_net, args.start_iteration_ensemble)
        self.classifier_average = Moving_average_net(self.task_specific_head, args.start_iteration_clf)

    
    # Define a method called load_data that loads the data for the model.
    def load_data(self, args, epoch = -1):
        
        # Check if args.dataset is equal to 'domainbed'
        if args.dataset == 'domainbed':
            
            # Set the data directory to './miscellaneous/domainbed/data/'
            data_dir = './miscellaneous/domainbed/data/'
            
            # Check if args.super_mixed is False
            if args.super_mixed == False:
                
                # Check if epoch is equal to -1
                if epoch == -1:
                    work_num_dic = {'DomainNet': 0, 'VLCS': 4, 'PACS': 4, 'OfficeHome': 2, 'TerraIncognita': 0}
                else:
                    work_num_dic = {'DomainNet': 2, 'VLCS': 4, 'PACS': 4, 'OfficeHome': 2, 'TerraIncognita': 4}
                
                # Load the data using the domainbed_dataloader function and assign it to the train_loader, val_loader, test_loader, full_loader, and class_name variables
                self.train_loader, self.val_loader, self.test_loader, self.full_loader, self.class_name = domainbed_dataloader(dataset = args.domainbed_dataset, data_dir = data_dir,\
                    test_envs = [args.domainbed_test_env], batch_size = args.batch_size,
                    domain_random = args.mixed_training, work_num = work_num_dic[args.domainbed_dataset])

                # Check if type(self.train_loader) is equal to list
                if type(self.train_loader) == list:
                    
                    # Set the steps_per_epoch variable to the minimum value of the length of the train_loader list
                    self.steps_per_epoch = min([len(self.train_loader[i]) for i in range(len(self.train_loader))])
                    
                    # Zip the train_loader list and assign the result to the train_loader variable
                    self.train_loader = zip(*self.train_loader)
                    
                    # Set the num_epoch variable to the integer division of 5001 and the steps_per_epoch
                    self.num_epoch = int(5001/self.steps_per_epoch)
                else:
                    
                    # Set the steps_per_epoch variable to the length of the train_loader
                    self.steps_per_epoch = len(self.train_loader)
                    
                    # Set the num_epoch variable to the integer division of 5001 and the steps_per_epoch
                    self.num_epoch = int(5001/self.steps_per_epoch)
            
            # Check if args.super_mixed is True
            else:
                
                # Check if epoch is equal to -1
                if epoch == -1:
                    work_num_dic = {'DomainNet': 0, 'VLCS': 4, 'PACS': 4, 'OfficeHome': 2, 'TerraIncognita': 0}
                else:
                    work_num_dic = {'DomainNet': 4, 'VLCS': 4, 'PACS': 4, 'OfficeHome': 2, 'TerraIncognita': 4}
                
                # Load the data using the domainbed_dataloader function and assign it to the train_loader_1, val_loader, test_loader, full_loader, and class_name variables
                train_loader_1, self.val_loader, self.test_loader, self.full_loader, self.class_name = domainbed_dataloader(dataset = args.domainbed_dataset, data_dir = data_dir,\
                    test_envs = [args.domainbed_test_env], batch_size = args.batch_size,
                    domain_random = False, work_num = work_num_dic[args.domainbed_dataset])
                
                # Load the data using the domainbed_dataloader function with domain_random set to True and assign the result to the train_loader_2 variable
                train_loader_2, self.val_loader, self.test_loader, self.full_loader, self.class_name = domainbed_dataloader(dataset = args.domainbed_dataset, data_dir = data_dir,\
                    test_envs = [args.domainbed_test_env], batch_size = args.batch_size,
                    domain_random = True, work_num = work_num_dic[args.domainbed_dataset])
                
                # Set the steps_per_epoch variable to the minimum value of the length of the train_loader_1 list
                self.steps_per_epoch = min([len(train_loader_1[i]) for i in range(len(train_loader_1))])
                
                # Check if the dataset is 'domainbed' and args.domainbed_dataset is 'DomainNet'
                if args.dataset == 'domainbed' and args.domainbed_dataset == 'DomainNet':
                    
                    # Set the steps_per_epoch variable to the integer division of steps_per_epoch and args.domainnet_split_point
                    self.steps_per_epoch /= args.domainnet_split_point
                
                # Set the steps_per_epoch to an integer
                self.steps_per_epoch = int(self.steps_per_epoch)
                
                # Zip the train_loader_1 list and assign the result to the train_loader variable
                train_loader_1 = zip(*train_loader_1)
                
                # Set the num_epoch variable to the integer division of 5001 and the steps_per_epoch
                self.num_epoch = int(5001/self.steps_per_epoch)
                
                # Set the train_loader variable to a tuple of train_loader_1 and train_loader_2
                self.train_loader = (train_loader_1, train_loader_2)
    
    # Define a forward method that takes in input, model_preds, split, labels, class_type, epoch, and envs arguments
    def forward(self, x_in, model_preds = None, split = None, labels = None, class_type = None, epoch = None, envs = None):
        
        # Get the class type from the dataset_inference_information dictionary
        class_type = dataset_inference_information[self.dataset][self.args.domainbed_dataset]['class_type']
        
        # Put the task specific head and ensemble net on the device
        self.task_specific_head.to(self.device)
        self.ensemble_net.to(self.device)
        
        # Create an empty list called agg
        agg = []
        
        # Loop through each model prediction and index using model_idx and y_hat
        for model_idx, y_hat in enumerate(model_preds):
            
            # Create an empty list called clf_out
            clf_out = []
            
            # Move the y_hat tensor to the device
            y_hat = y_hat.to(self.device)

            # Check if the length of the y_hat tensor shape is equal to 4
            if len(y_hat.shape) == 4:
                
                # Squeeze the y_hat tensor using the last two dimensions and assign the result to the y_hat variable
                y_hat = y_hat.squeeze(3).squeeze(2)

            # Check if the last dimension of the y_hat tensor is equal to class_type
            if y_hat.squeeze().shape[-1] == class_type:
                
                # Assign y_hat to the ori_out variable
                ori_out = y_hat
            else:
                idx_task_specific_head = self.task_head_list.index(y_hat.squeeze().shape[-1])
                
                # Use the task specific head indexed by idx_task_specific_head to compute the output using the y_hat tensor and split tensor and assign the result to the ori_out variable
                ori_out = self.task_specific_head[idx_task_specific_head](y_hat, split)

            # Append the ori_out tensor to the clf_out list as an unsqueezed tensor
            clf_out.append(ori_out.unsqueeze(1))
            
            # Concatenate the tensors in clf_out along the second dimension and append the result to agg
            agg.append(torch.cat(clf_out, dim = 1))

        # Concatenate the tensors in agg along the third dimension and assign the variable to y_hat
        y_hat = torch.cat(agg, dim = 2).to(self.device)
        
        # Send x_in, y_hat, split, labels, class_type, envs, and self.args to the ensemble_net to compute the output, child_loss, confidence_loss, ensemble_loss, weighted_mat, and true_confs and assign the results to output, child_loss, confidence_loss, ensemble_loss, weighted_mat, and true_confs
        output, child_loss, confidence_loss, ensemble_loss, weighted_mat, true_confs = self.ensemble_net(x_in, y_hat, split, labels, class_type, envs, self.args)

        # Check if args.ensemble_mode is equal to 'dot_product'
        if self.args.ensemble_mode == 'dot_product':
            # Compute the loss using the ensemble_loss, child_loss, confidence_loss, and assign it to loss
            loss = self.args.ensemble_weight * ensemble_loss + self.args.child_weight * child_loss + self.args.confidence_weight * confidence_loss
        
        elif self.args.ensemble_mode == 'mean' or self.args.ensemble_mode == 'pure_model_adapter':
            # Assign child_loss to loss
            loss = child_loss


        bp_start_time = time.time()
        
        # Zero the optimizer gradients for the ensemble net
        self.optimizer_ensemble.zero_grad()
        # Check if args.joint_optimizer is False
        if not self.args.joint_optimizer:
            # Zero the optimizer gradients for the classifier
            self.optimizer_clf.zero_grad()
            
        # Compute the gradients for the loss tensor
        loss.backward()
        
        # Take a step using the optimizer for the ensemble net
        self.optimizer_ensemble.step()

        if not self.args.joint_optimizer:
            # Take a step using the optimizer for the classifier
            self.optimizer_clf.step()
        
        # Compute the amount of time taken to do backpropagation and assign it to bp_time
        bp_time = time.time() - bp_start_time

        if epoch == 0 and self.args.warmup_epoch != 0:
            # Step the warmup scheduler
            self.warmup_scheduler.step()
            if not self.args.joint_optimizer:
                self.clf_warmup_scheduler.step()       
        else:
            # Step the ensemble scheduler
            self.scheduler_ensemble.step()
            if not self.args.joint_optimizer:
                # Step the classifier scheduler
                self.scheduler_clf.step()

        if self.dataset == 'domainbed':
            # Update the ensemble average
            self.ensemble_average.update_sma()
            # Update the classifier average
            self.classifier_average.update_sma()

        # Return output, loss, a list of tensors resulting from summing agg along the second dimension and then squeezing the result along the first dimension, child_loss, confidence_loss, ensemble_loss, and bp_time
        return output, loss, [torch.sum(item, dim = 1).squeeze() for item in agg], child_loss, confidence_loss, ensemble_loss, bp_time


    def predict(self, x_in, model_preds = None, split = None, labels = None, class_type = None, args = None, ensemble_mode = 'dot_product', batch_index = 0):

        # The below line re-defines the class_type based on 'dataset_inference_information' list and the 'args' object passed to the 'predict' function.
        class_type = dataset_inference_information['domainbed'][args.domainbed_dataset]['class_type']

        # The below lines move the 'network_sma' module of 'classifier_average' and 'ensemble_average' objects to the device available for computation.
        # Then, it sets the 'eval' mode for the 'network_sma' of all objects.
        # 'task_specific_head' and 'ensemble_net' are also set in the 'eval' mode.
        self.classifier_average.network_sma.to(self.device)
        self.ensemble_average.network_sma.to(self.device)
        self.classifier_average.network_sma.eval()
        self.ensemble_average.network_sma.eval()
        self.task_specific_head.eval()
        self.ensemble_net.eval()

        # The empty 'agg' list is initialized.
        agg = []
        
        # The following 'for' loop iterates over the model predictions in the 'model_preds' list.
        for _, y_hat in enumerate(model_preds):
            clf_out = []
            y_hat = y_hat.to(self.device)

            # Checks the 'y_hat' tensor dimensions and resizes the tensor based on conditions specified in 'if' loops.
            if len(y_hat.shape) == 4:
                y_hat = y_hat.squeeze(3).squeeze(2)

            # In this 'if' loop, it checks if the last dimension of 'y_hat' is equal to 'class_type'.
            # If yes, then it assigns 'y_hat' to 'ori_out' variable.
            # If not, it assigns the specific head corresponding to the 'y_hat' tensor to 'ori_out'.
            if y_hat.squeeze().shape[-1] == class_type:
                ori_out = y_hat
            else:
                idx_task_specific_head = self.task_head_list.index(y_hat.squeeze().shape[-1])
                ori_out = self.task_specific_head[idx_task_specific_head](y_hat, split)

            # The 'ori_out' tensor is appended to the 'clf_out' list by unsqeezing at dim=1.
            clf_out.append(ori_out.unsqueeze(1))

            # The 'clf_out' list is concatenated using 'torch.cat' function at dim=1, and finally to the 'agg' list.
            agg.append(torch.cat(clf_out, dim = 1))

        # All the tensors in 'agg' list are concatenated along dim=2 and moved to the computation device.
        y_hat = torch.cat(agg, dim = 2).to(self.device)

        # The following if-else statements execute if the 'args.average_ensemble' flag is True or False respectively.
        if args.average_ensemble:
            # If the flag is True, the 'network_sma' of 'ensemble_average' is called with parameters like x_in, y_hat, split, labels, class_type, and args.
            output, child_loss, confidence_loss, ensemble_loss, weighted_mat, true_confs = self.ensemble_average.network_sma(x_in, y_hat, split, labels, class_type, args = args)
        else:
            # If the flag is False, the 'ensemble_net' is called with same parameters as above.
            output, child_loss, confidence_loss, ensemble_loss, weighted_mat, true_confs = self.ensemble_net(x_in, y_hat, split, labels, class_type, args = args)

        # The 'ensemble_loss', 'child_loss', and 'confidence_loss' variables are added to compute the 'loss'.
        loss = ensemble_loss + child_loss + confidence_loss

        # If 'ensemble_mode' is 'pure_model_adapter' and 'split' is 'evaluation', it saves y_hat, labels, weighted_mat, and true_confs as '.pth' files.
        if ensemble_mode == 'pure_model_adapter' and split == 'evaluation':
            torch.save(y_hat, os.environ.get('AMLT_OUTPUT_DIR', '.') + '/' + str(args.domainbed_test_env) + 'y_hat_' + str(batch_index) +  ".pth")
            torch.save(labels, os.environ.get('AMLT_OUTPUT_DIR', '.') + '/' + str(args.domainbed_test_env) + 'labels_' + str(batch_index) +  ".pth")
            torch.save(weighted_mat, os.environ.get('AMLT_OUTPUT_DIR', '.') + '/' + str(args.domainbed_test_env) + 'weighted_mat_' + str(batch_index) +  ".pth")
            torch.save(true_confs, os.environ.get('AMLT_OUTPUT_DIR', '.') + '/' + str(args.domainbed_test_env) + 'true_confs_' + str(batch_index) +  ".pth")

        # Returns the output, loss and a list containing tensors with sums along dim=1 and then squeezed along dim=1 for all tensors in the 'agg' list.
        return output, loss, [torch.sum(item, dim = 1).squeeze(1) for item in agg], child_loss, confidence_loss, ensemble_loss


class Ensemble_net(torch.nn.Module):
    
    # Initialize the neural network
    def __init__(self, args):
        
        # Call the constructor of the parent class
        super(Ensemble_net, self).__init__()

        # Assign values to neural network parameters
        self.num_local = args.num_local
        self.num_model = args.model_num
        self.child_attach_ensembleloss = True
        self.model_num = args.model_num
        self.mode = args.ensemble_mode
        self.use_true_conf = args.use_true_conf
        self.use_optimal = args.use_optimal
        self.args = args

        # Create a parameter tensor with model_num rows and 1000 columns
        self.model_emb = torch.nn.Parameter(torch.ones((self.model_num, 1000)), requires_grad=True)
        # Initialize the tensor with random values using xavier_uniform distribution
        nn.init.xavier_uniform_(self.model_emb)

        if args.use_advanced_extractor:
            # Use a pre-trained CNN model from timm library with the specified number of output classes
            self.feature_extractor = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes = 0)
            img_emb_dim = 2560
        else:
            # Use pre-trained resnet34 model
            self.feature_extractor = models.resnet34(pretrained=True)
            img_emb_dim = 1000

        # Freeze the layers to avoid updating pre-trained weights
        freeze(self.feature_extractor)

        if args.dataset == 'domainbed':
            self.source_domain_num = args.domain_num - 1

        self.target_domain = args.domainbed_test_env

        # Define image embedding size
        embedding_size = 100
        # Define fully-connected layers with the specified input and output dimensions
        self.w1 = torch.nn.Linear(img_emb_dim, embedding_size)
        self.w2 = torch.nn.Linear(1000, embedding_size)
        self.w3 = torch.nn.Linear(self.model_num, self.model_num)

        # Define loss functions for classification and confidence
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.loss_func_reduce = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.confidence_loss_func = torch.nn.BCEWithLogitsLoss()
        self.ensemble_loss_func = torch.nn.CrossEntropyLoss()

        # Define rectified linear unit activation function
        self.m = torch.nn.ReLU()
    
    # Forward propagation method
    def forward(self, x_in, y_pred = None, split = None, labels = None, class_type = None, envs = None, args = None):
        
        # If using dot product as the ensemble mode
        if self.mode == 'dot_product':

            # Extract image features and apply a ReLU activation function
            x_in = self.m(self.w1(self.feature_extractor(x_in))) # [B,1000]
            # Apply a ReLU activation function to the model embedding tensor
            model_emb = self.m(self.w2(self.model_emb)) # [num_model,1000]
            # Compute the weights using the dot product of input and model embeddings and apply a softplus operation
            weights = F.softplus(self.w3(torch.matmul(x_in, model_emb.t()))) # [B,num_model]
            # Squeeze y_pred tensor
            y_pred = y_pred.squeeze(1)

        take_post_processed_weight = False

        # Concatenate softmax probabilities into a single tensor
        softmax_multi_child_pred = torch.cat([torch.softmax(y_pred[:, model_index * class_type: (model_index+1) * class_type], dim = -1) for model_index in range(self.model_num)], dim = 1)
        softmax_multi_child_pred = softmax_multi_child_pred.squeeze(1)

        # Compute the true confidence scores by concatenating softmax probabilities along the class axis and applying a softmax function
        true_confs = torch.softmax(torch.cat([softmax_multi_child_pred[index,true_class::class_type].unsqueeze(0) for index, true_class in enumerate(labels)], dim = 0).detach(), dim = 1)

        # If using true confidence scores
        if self.use_true_conf:
            weights = true_confs

        # If the ensemble mode is mean and neural network is in training mode or 
        # ensemble mode is not random or pure model adapter
        if self.mode == 'mean':
            if self.training:
                # Set weights to ones
                weights = torch.ones_like(true_confs)
            else:
                # Otherwise, set weights to zeros
                weights = torch.zeros_like(true_confs)
                sample = np.random.choice(range(self.model_num), (weights.shape[0],self.num_local))
                for row_idx, row in enumerate(sample):
                    for col in row:
                        weights[row_idx, col] = 1
        elif self.mode == 'pure_model_adapter':
            # Set weights to ones
            weights = torch.ones_like(true_confs)

        # If the number of local models is less than the total number of models
        if self.num_local < self.model_num:
            post_weights = torch.zeros_like(true_confs)
            take_post_processed_weight = True
            sample = np.array(weights.topk(self.num_local)[1].cpu())
            for row_idx, row in enumerate(sample):
                for col in row:
                    post_weights[row_idx, col] = weights[row_idx, col]

            # Normalize post-processed weights
            post_weights = F.normalize(post_weights, p=1, dim=1, eps=1e-12)

        # Normalize weights
        weighted_mat = F.normalize(weights, p=1, dim=1, eps=1e-12)

        # If using optimal weights
        if self.use_optimal:
            weighted_mat = torch.zeros_like(weights)
            # Find the highest true confidence scores
            optimal_idx = torch.topk(true_confs,1).indices
            for i in range(weights.shape[0]):
                # Set the corresponding weight to 1
                weighted_mat[i,optimal_idx[i][0]] = 1

        # Squeeze y_pred tensor
        y_pred = y_pred.squeeze(1)
        # If detached child
        if args.detach_child:
            ems_out = torch.cat([torch.sum(y_pred.detach()[:,i::class_type] * weighted_mat, dim=1).unsqueeze(1) for i in range(class_type)], dim = 1)
            if take_post_processed_weight:
                ems_out_post = torch.cat([torch.sum(y_pred.detach()[:,i::class_type] * post_weights, dim=1).unsqueeze(1) for i in range(class_type)], dim = 1)
        else:
            ems_out = torch.cat([torch.sum(y_pred[:,i::class_type] * weighted_mat, dim=1).unsqueeze(1) for i in range(class_type)], dim = 1)
            
            if take_post_processed_weight:
                ems_out_post = torch.cat([torch.sum(y_pred[:,i::class_type] * post_weights, dim=1).unsqueeze(1) for i in range(class_type)], dim = 1)

        each_point_loss = torch.ones((labels.shape[0],self.model_num)).to(x_in.device)
        for j in range(self.model_num):   
            each_point_loss[:,j] = self.loss_func_reduce(y_pred[:,j*class_type:(j+1)*class_type], labels).to(x_in.device)

        # If the mode is other than random or pure model adapter
        if self.mode not in ['random', 'pure_model_adapter']:
            # Compute confidence loss
            confidence_loss = self.confidence_loss_func(torch.nn.functional.softmax(weighted_mat, dim = -1),
                torch.nn.functional.softmax(true_confs, dim = -1))
        else:
            confidence_loss = 0

        # Normalize weight tensor
        weighted_mat = torch.nn.functional.softmax(weighted_mat, dim = -1)

        # If local weighting is not used
        if args.no_local_weighting:
            # Compute the child loss by taking the product of each point's loss with a tensor of ones
            child_loss = torch.mean(torch.mul(each_point_loss, torch.ones_like(weighted_mat)))
        else:
            # Compute the child loss by taking the product of each point's loss with the normalized weight
            child_loss = torch.mean(torch.mul(each_point_loss, weighted_mat)) # weighted_mat + local_init_weight

        # Compute ensemble loss
        ensemble_loss = self.ensemble_loss_func(ems_out, labels)

        # Return appropriate tensor values depending on whether post-processed weights are taken or not
        if take_post_processed_weight:
            return ems_out_post, child_loss, confidence_loss,\
                ensemble_loss, weighted_mat, true_confs
        else:
            return ems_out, child_loss, confidence_loss,\
                ensemble_loss, weighted_mat, true_confs



# Defining a class 'task_head'
# which inherits from the PyTorch base class 'torch.nn.Module'
class task_head(torch.nn.Module):
    
    # Constructor of the class
    # It takes input_dim and out_dim as parameters
    def __init__(self, input_dim, out_dim):
        
        # Calling the constructor of parent class 'torch.nn.Module'
        super(task_head, self).__init__()
        
        # Assigning input_dim and out_dim as instance variables for class 'task_head'
        self.out_dim = out_dim
        self.input_dim = input_dim
        
        # Defining a linear layer for encoding the input tensor
        # which will be used for classification
        self.encoder_2 = torch.nn.Linear(input_dim, out_dim)
    
    # Defining the 'forward' method which contains the network logic
    def forward(self, out, split = None):
        
        # If output dimension is not 200
        if self.out_dim != 200:
            
            # If number of dimensions of 'out' tensor is 4
            if len(out.shape) == 4:
                # Squeezing out the last two dimensions
                out = out.squeeze(3).squeeze(2)
            
            # Encoding the 'out' tensor using the linear layer
            out = self.encoder_2(out)
        else:
            # If split parameter is set to 'evaluation'
            if split == 'evaluation':
                # Selecting subset of indices from the 'out' tensor 
                # for the evaluation split
                out = out[:,self.indices_in_1k]
        
        # Returning the encoded tensor
        return out

# Defining a function 'freeze'
# which takes a layer as an input
def freeze(layer):
    
    # Looping through each child layer of the input layer
    for child in layer.children():
        
        # Looping through each parameter of each child layer
        for param in child.parameters():
            # Setting the parameter's 'requires_grad' attribute to False
            # to freeze it's learning during training
            param.requires_grad = False