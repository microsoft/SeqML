# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import copy
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from torchensemble._base import BaseClassifier
from torchensemble._base import torchensemble_model_doc
from torchensemble.utils import io
from torchensemble.utils import set_module
from torchensemble.utils import operator as op
from utils import WarmUpLR, accuracy, AverageMeter, ProgressMeter, ThresholdBinarizer, cost_calculation


__all__ = ["EnsembleClassifier"]
dis_criterion = torch.nn.L1Loss(reduce=False)
look_up_map = [[] for _ in range(50001)]


def subset_ensemble_evaluate(cls1_old, cls2_old, distill_old, cls_aux_old, selectors, current_ens, args, current_exit_mask = None):
    """ Evaluate the ensemble of all activated models at a certain timestep """

    exit_mask_before = torch.ones(cls1_old.size(0)).cuda()
    avg_ens, earlyexit_out, masks = [], [], []
    Exit_rate_l = []
    exit_mask_stacked = []
    
    for idx, _ in enumerate(range(cls1_old.size(1))):

        cls1, cls2, distill_out, cls_aux = cls1_old[:,idx,:], cls2_old[:,idx,:], distill_old[:,idx,:], cls_aux_old[:,idx,:]
        cls_aux = (cls1 + cls2) / 2

        ens = F.softmax(cls_aux * (1 - args.alpha) + distill_out * args.alpha)
        ens = ens.unsqueeze(0)
        avg_ens.append(ens)
        
        mask = selectors(cls1_old, cls2_old, cls_aux_old, distill_old, idx = idx)
        mask_ = mask.unsqueeze(0).unsqueeze(2)
        mask_ = mask_.expand(ens.shape)
        masks.append(mask_)

        if idx == 0:
            exit_rate = torch.sum(mask, dtype=torch.float32) / mask.size(0) * 100.0
        else:
            mask_for_use = exit_mask_before.unsqueeze(0).unsqueeze(2).expand(ens.shape)
            ens = ens*mask_for_use + earlyexit_out[-1]*(-mask_for_use+1)
            mask_increase = exit_mask_before*mask
            exit_rate = torch.sum(mask_increase, dtype=torch.float32) / mask.size(0) * 100.0

        Exit_rate_l.append(exit_rate)
        exit_mask_before = exit_mask_before * (-mask+1)
        exit_mask_stacked.append(exit_mask_before.unsqueeze(0))
        earlyexit_out.append(ens)

    current_mask_increase = exit_mask_before*current_exit_mask
    current_exit_rate = torch.sum(current_mask_increase, dtype=torch.float32) / mask.size(0) * 100.0
    Exit_rate_l.append(current_exit_rate)

    cost = cost_calculation(Exit_rate_l, current_ens is None, args.n_estimators)
    ens_old = torch.cat(avg_ens, dim = 0)
    avg_ens = torch.mean(torch.cat(avg_ens, dim = 0), dim=0)
    exit_mask_stacked = torch.cat(exit_mask_stacked, dim = 0)
    ens_old_result = torch.mean(ens_old, dim=0)
    mask_for_use = exit_mask_before.unsqueeze(1).expand(ens_old_result.shape)
    current_ens_for_selectors = current_ens.detach()*mask_for_use + ens_old_result.squeeze()*(-mask_for_use+1)
    current_ens_for_models = current_ens*mask_for_use.detach() + ens_old_result.squeeze()*(-mask_for_use+1).detach()
    return ens_old, exit_mask_before, current_ens_for_selectors, current_ens_for_models, cost, avg_ens, exit_mask_stacked


class Selector(torch.nn.Module):

    def __init__(self, num_class, args):

        super(Selector, self).__init__()

        hid_dim = 2
        input_size = 2
        self.args = args
        self.num_class = num_class
        self.n_state = args.n_estimators
        self.margin = torch.ones([1])/self.n_state
        self.network = nn.LSTM(
            input_size=input_size,
            hidden_size=hid_dim,
            num_layers=1,
            batch_first=True,
        )
        self.init_h = nn.Parameter(torch.zeros(1, hid_dim))
        self.init_c = nn.Parameter(torch.zeros(1, hid_dim))

        for p in self.network.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, cls1, cls2, cls_aux = None, distill_out = None, idx = None):
        _, pred_1 = cls1[:,idx,:].topk(1, 1, True, True)
        _, pred_2 = cls2[:,idx,:].topk(1, 1, True, True)
        exit_mask = pred_1.view(-1) == pred_2.view(-1)

        if cls_aux != None and distill_out != None:
            _, pred_aux = cls_aux[:,idx,:].topk(1, 1, True, True)
            _, pred_distill = distill_out[:,idx,:].topk(1, 1, True, True)
            exit_mask = exit_mask.__and__(pred_aux.view(-1) == pred_distill.view(-1))

        input_to_lstm_ = []

        for i in range(2):
            if i == 0:
                network_input1 = [cls1]
                network_input2 = [cls2]
            else:
                network_input1 = [cls_aux]
                network_input2 = [distill_out]  
                                      
            for _ in range(idx+1, self.n_state):
                network_input1.append(torch.zeros_like(cls1[:,0:1,:]))
                network_input2.append(torch.zeros_like(cls2[:,0:1,:]))
                
            network_input1 = torch.cat(network_input1, dim = 1)
            network_input2 = torch.cat(network_input2, dim = 1)
            
            kl = nn.KLDivLoss(reduction="none", log_target=False)
            
            soft_argmax_1 = F.gumbel_softmax(network_input1, tau=1, hard=False)
            soft_argmax_2 = F.gumbel_softmax(network_input2, tau=1, hard=False)
            kl_loss = kl(soft_argmax_1, soft_argmax_2)
            input_to_lstm = torch.sum(kl_loss, dim = -1).unsqueeze(-1)
            input_to_lstm_.append(input_to_lstm)
            
        input_to_lstm = torch.cat(input_to_lstm_, dim = -1)
        state_size = (self.init_h.shape[0], input_to_lstm.shape[0], self.init_h.shape[1])
        init_h = self.init_h.expand(*state_size).contiguous()
        init_c = self.init_c.expand(*state_size).contiguous()
        cls_cat, _ = self.network(input_to_lstm, (init_h, init_c))
        cls_cat  = F.gumbel_softmax(cls_cat, tau=1, hard=False, dim = -1)
        cls_cat = cls_cat[:,idx,:]
        
        return ThresholdBinarizer.apply(cls_cat[:,1], self.margin.to(cls_cat.device), False).squeeze()


def _parallel_fit_per_epoch(
    train_loader,
    estimator,
    cur_lr,
    optimizer,
    criterion,
    idx,
    epoch,
    log_interval,
    device,
    logger,
    args,
    writer,
    selectors,
    selector_optimizers,
    selector_lr
):

    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """

    r_loss_m = nn.MarginRankingLoss(reduction='none', margin=args.ranking_loss_margin_m)
    r_loss_s = nn.MarginRankingLoss(reduction='none', margin=args.ranking_loss_margin_s)

    if cur_lr:
        # Parallelization corrupts the binding between optimizer and scheduler
        set_module.update_lr(optimizer, cur_lr)
    if selector_lr:
        set_module.update_lr(selector_optimizers, selector_lr)

    if epoch < args.warm:
        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    estimator_now = estimator[idx]
    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data_time', ':6.3f')
    Train_loss = AverageMeter('Train_loss', ':6.3f')
    Loss_cls_1 = AverageMeter('Loss_cls_1', ':6.3f')
    Loss_cls_2 = AverageMeter('Loss_cls_2', ':6.3f')
    Loss_dis = AverageMeter('Loss_dis', ':6.3f')
    Loss_ranking = AverageMeter('Loss_ranking', ':6.3f')
    Loss_ensemble = AverageMeter('Loss_ensemble', ':6.3f')
    Loss_cost = AverageMeter('Loss_cost', ':6.3f')
    Loss_distill = AverageMeter('Loss_distill', ':6.3f')
    Aux_Head1_Acc1 = AverageMeter('Aux_Head1_Acc', ':6.2f')
    Aux_Head2_Acc1 = AverageMeter('Aux_Head2_Acc', ':6.2f')
    Acc_aux_ens = AverageMeter('Acc_aux_ens', ':6.2f')
    Acc_distill = AverageMeter('Acc_distill', ':6.2f')
    Acc_ens = AverageMeter('Acc_ens', ':6.2f')
    Acc_ens_sf = AverageMeter('Acc_ens_sf', ':6.2f')
    Acc_same = AverageMeter('Acc_same', ':6.2f')
    Acc_diff = AverageMeter('Acc_diff', ':6.2f')
    Acc_all = AverageMeter('Acc_all', ':6.2f')
    Exit_rate = AverageMeter('Exit_rate', ':6.2f')
    Overall_exit_rate = AverageMeter('Overall_exit_rate', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [Batch_time, Data_time, Train_loss, Loss_cls_1, Loss_cls_2, Loss_dis, Loss_ranking, Loss_ensemble, Loss_cost, Loss_distill, Exit_rate, Overall_exit_rate, Acc_same, Acc_diff, Acc_all, Aux_Head1_Acc1, Aux_Head2_Acc1, Acc_aux_ens, Acc_distill, Acc_ens, Acc_ens_sf],
        prefix=f"Epoch: [{epoch}] Ens:[{idx+1}] Lr:[{optimizer.state_dict()['param_groups'][0]['lr']:.5f}] Selector_Lr:[{selector_optimizers.state_dict()['param_groups'][0]['lr']:.5f}]",
    logger=logger)
    end = time.time()

    for batch_idx, elem in enumerate(train_loader):

        Data_time.update(time.time() - end)
        data_id, data, target = io.split_data_target(elem, device)
        batch_size = target.size(0)

        optimizer.zero_grad()
        selector_optimizers.zero_grad()

        cls1, cls2, distill_out = estimator_now(*data)
        cls_aux = (cls1 + cls2) / 2

        ens = F.softmax(cls_aux * (1 - args.alpha) + distill_out * args.alpha)
        ens_sf = F.softmax((F.softmax(cls1, 1) + F.softmax(cls2, 1)) * (1 - args.alpha)  + F.softmax(distill_out, 1) * args.alpha)

        acc_1, _ = accuracy(cls1, target)
        acc_2, _ = accuracy(cls2, target)
        acc_aux_ens, _ = accuracy(cls_aux, target)
        acc_distill, _ = accuracy(distill_out, target)
        acc_ens, _ = accuracy(ens, target)
        acc_ens_sf, _ = accuracy(ens_sf, target)

        if idx == 0:
            exit_mask = selectors(cls1.unsqueeze(1), cls2.unsqueeze(1), cls_aux.unsqueeze(1), distill_out.unsqueeze(1), idx)
        else:
            _, _, _, cls1_old, cls2_old, distillout_old, cls_aux_old = look_up(data_id)
            cls1_all = torch.cat([cls1_old, cls1.unsqueeze(1)], dim = 1)
            cls2_all = torch.cat([cls2_old, cls2.unsqueeze(1)], dim = 1)
            distill_out_all = torch.cat([distillout_old, distill_out.unsqueeze(1)], dim = 1)
            cls_aux_all = torch.cat([cls_aux_old, cls_aux.unsqueeze(1)], dim = 1)
            exit_mask = selectors(cls1_all, cls2_all, cls_aux_all, distill_out_all, idx)

        exit_rate = torch.sum(exit_mask, dtype=torch.float32) / batch_size * 100.0
        acc_same, _ = accuracy(ens[exit_mask.type(torch.bool)], target[exit_mask.type(torch.bool)]) if target[exit_mask.type(torch.bool)].size(0) > 0 else (ens.new_tensor([0]), 0)
        acc_diff, _ = accuracy(ens[~exit_mask.type(torch.bool)], target[~exit_mask.type(torch.bool)]) if target[~exit_mask.type(torch.bool)].size(0) > 0 else (ens.new_tensor([0]), 0)
        acc_all, _ = accuracy(ens, target)

        """L_cls for training accurate base models"""
        loss_cls_1 = criterion(cls1, target)
        loss_cls_2 = criterion(cls2, target)
        cls_loss = (loss_cls_1 + loss_cls_2) / 2

        """L_dis for training diverse base models to estimate sample hardness"""
        dis_loss = - torch.mean(dis_criterion(F.softmax(cls1, 1), F.softmax(cls2, 1)), dim=-1) * args.aux_dis_lambda

        aux_inner_distillation_loss = torch.mean(criterion(distill_out, target)) * args.alpha

        if idx != 0:
            ens_old, exit_mask_before, current_ens_for_selectors, current_ens_for_models, cost, avg_ens_pre, exit_mask_stacked = subset_ensemble_evaluate(cls1_old.detach(), cls2_old.detach(), distillout_old.detach(), cls_aux_old.detach(), selectors, ens, args, exit_mask)

            loss = cls_loss + dis_loss 
            loss = torch.mean(loss)

            """Ranking loss, which is used to update K-th model and the selector"""
            # calculate loss of previous (k-1)-model ensemble results
            ens_pre = torch.mean(ens_old, dim=0)
            loss_cls_for_pre_ens = criterion(ens_pre, target) # Only params of selectors are active for optimization

            if exit_mask_before.requires_grad is False:
                mask_for_ranking = target.new_tensor([-1]).expand(target.size(0)) * exit_mask_before.int() + target.new_tensor([1]).expand(target.size(0)) * (-(exit_mask_before).int() + 1)
            else:
                mask_for_ranking = target.new_tensor([-1]).expand(target.size(0)) * exit_mask_before + target.new_tensor([1]).expand(target.size(0)) * (-exit_mask_before + 1)

            ranking_loss_for_model = r_loss_m(criterion(distill_out, target), loss_cls_for_pre_ens.detach(), mask_for_ranking.detach())
            ranking_loss_for_model = torch.mean(ranking_loss_for_model * exit_mask_before)
            ranking_loss_for_selectors = target.new_tensor([0.0], dtype = torch.float)

            for pre_idx in range(ens_old.shape[0]):
                loss_cls_for_pre_ens_ = criterion(torch.mean(ens_old[:pre_idx+1,:,:], dim=0), target)
                if exit_mask_stacked.requires_grad is False:
                    mask_for_ranking_ = target.new_tensor([-1]).expand(target.size(0)) * exit_mask_stacked[pre_idx,:].int() + target.new_tensor([1]).expand(target.size(0)) * (-(exit_mask_stacked[pre_idx,:]).int() + 1)
                else:
                    mask_for_ranking_ = target.new_tensor([-1]).expand(target.size(0)) * exit_mask_stacked[pre_idx,:] + target.new_tensor([1]).expand(target.size(0)) * (-exit_mask_stacked[pre_idx,:]+1)

                if pre_idx + 1 == idx:
                    cls_loss_ = criterion(distill_out.detach(), target)
                else:
                    cls_loss_ = criterion(distillout_old[:,pre_idx+1,:].detach(), target)

                ranking_loss_for_each_layer = r_loss_s(cls_loss_.detach(), loss_cls_for_pre_ens_.detach(), mask_for_ranking_)
                ranking_loss_for_each_layer = torch.mean(ranking_loss_for_each_layer * exit_mask_stacked[pre_idx,:].int()) 
                ranking_loss_for_selectors += ranking_loss_for_each_layer

            ranking_loss = ranking_loss_for_selectors * args.use_ranking_loss_s + ranking_loss_for_model * args.use_ranking_loss_m
            ranking_loss = torch.mean(ranking_loss)

            """Cost loss, for optimizing selectors only"""
            cost_loss = cost * args.use_cost_loss

            """Ensemble loss for training selectors"""
            ensemble_loss = torch.mean(criterion(current_ens_for_selectors, target) * args.use_ensemble_loss)

            loss = loss + ranking_loss + cost_loss + ensemble_loss + aux_inner_distillation_loss

        else:
            """Cost loss, for optimizing selectors only"""
            cost = cost_calculation([exit_rate], False, args.n_estimators)
            cost_loss = cost * args.use_cost_loss

            loss = cls_loss + dis_loss
            loss = torch.mean(loss) + aux_inner_distillation_loss

        loss_cls_1 = torch.mean(loss_cls_1)
        loss_cls_2 = torch.mean(loss_cls_2)
        dis_loss = torch.mean(dis_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=estimator_now.parameters(), max_norm=10, norm_type=2)
        torch.nn.utils.clip_grad_norm_(parameters=selectors.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        selector_optimizers.step()

        with torch.no_grad():
            Train_loss.update(loss.item(), batch_size)
            Loss_cls_1.update(loss_cls_1.item(), batch_size)
            Loss_cls_2.update(loss_cls_2.item(), batch_size)
            Loss_dis.update(dis_loss.item(), batch_size)
            Loss_distill.update(aux_inner_distillation_loss.item(), batch_size)

            if idx != 0:
                Loss_ranking.update(ranking_loss.item(), batch_size)
                Loss_ensemble.update(ensemble_loss.item(), batch_size)
                Loss_cost.update(cost_loss.item(), batch_size)
                Overall_exit_rate.update(cost.item(), batch_size)

            Exit_rate.update(exit_rate.item(), batch_size)
            Aux_Head1_Acc1.update(acc_1[0].item(), batch_size)
            Aux_Head2_Acc1.update(acc_2[0].item(), batch_size)
            Acc_aux_ens.update(acc_aux_ens[0].item(), batch_size)
            Acc_distill.update(acc_distill[0].item(), batch_size)

            Acc_ens.update(acc_ens[0].item(), batch_size)
            Acc_same.update(acc_same[0].item(), batch_size)
            Acc_diff.update(acc_diff[0].item(), batch_size)
            Acc_all.update(acc_all[0].item(), batch_size)
            Acc_ens_sf.update(acc_ens_sf[0].item(), batch_size)
            Batch_time.update(time.time() - end)
            end = time.time()

        # Print training status
        if batch_idx % log_interval == 0:
            progress.display(batch_idx)
            for k, v in progress.compose_json().items():
                writer.add_scalar(f"{k}/train_{idx+1}", v, epoch)
            writer.flush()

        if epoch < args.warm:
            warmup_scheduler.step()

    return estimator_now, optimizer, selector_optimizers


@torch.no_grad()
def _parallel_test_per_epoch(
    train_loader,
    estimator,
    idx,
    epoch,
    device,
    logger,
    args,
    writer,
    split,
    selectors
):
    """
    Private function used to test base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """

    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data_time', ':6.3f')
    Train_loss = AverageMeter('Train_loss', ':6.3f')
    Loss_cls_1 = AverageMeter('Loss_cls_1', ':6.3f')
    Loss_cls_2 = AverageMeter('Loss_cls_2', ':6.3f')
    Loss_cls_ens = AverageMeter('Loss_cls_ens', ':6.3f')
    Loss_distill = AverageMeter('Loss_distill', ':6.3f')
    Loss_dis = AverageMeter('Loss_dis', ':6.3f')
    Aux_Head1_Acc1 = AverageMeter('Aux_Head1_Acc', ':6.2f')
    Aux_Head2_Acc1 = AverageMeter('Aux_Head2_Acc', ':6.2f')
    Acc_aux_ens = AverageMeter('Acc_aux_ens', ':6.2f')
    Acc_same = AverageMeter('Acc_same', ':6.2f')
    Acc_diff = AverageMeter('Acc_diff', ':6.2f')
    Acc_all = AverageMeter('Acc_all', ':6.2f')
    Exit_rate = AverageMeter('Exit_rate', ':6.2f')
    Acc_ens = AverageMeter('Acc_ens', ':6.2f')
    Acc_ens_sf = AverageMeter('Acc_ens_sf', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [Batch_time, Data_time, Train_loss, Loss_cls_1, Loss_cls_2, Loss_cls_ens, Loss_dis, Loss_distill, Exit_rate, Acc_same, Acc_diff, Acc_all, Aux_Head1_Acc1, Aux_Head2_Acc1, Acc_aux_ens, Acc_ens, Acc_ens_sf],
        prefix=f"Epoch: [{epoch}] Ens:[{idx+1}]",
    logger=logger)

    end = time.time()
    for _, elem in enumerate(train_loader):
        Data_time.update(time.time() - end)

        data_id, data, target = io.split_data_target(elem, device)
        batch_size = target.size(0)

        cls1, cls2, distill_out = estimator(*data)
        cls_aux = (cls1 + cls2) / 2

        ens = F.softmax(cls_aux * (1 - args.alpha) + distill_out * args.alpha)
        ens_sf = F.softmax((F.softmax(cls1, 1) + F.softmax(cls2, 1)) * (1 - args.alpha)  + F.softmax(distill_out, 1) * args.alpha)

        acc_1, _ = accuracy(cls1, target)
        acc_2, _ = accuracy(cls2, target)
        acc_aux_ens, _ = accuracy(cls_aux, target)
        acc_ens, _ = accuracy(ens, target)
        acc_ens_sf, _ = accuracy(ens_sf, target)

        if idx == 0:
            exit_mask = selectors(cls1.unsqueeze(1), cls2.unsqueeze(1), cls_aux.unsqueeze(1), distill_out.unsqueeze(1), idx)
        else:
            _, _, _, cls1_old, cls2_old, distillout_old, cls_aux_old = look_up(data_id)
            cls1_all = torch.cat([cls1_old, cls1.unsqueeze(1)], dim = 1)
            cls2_all = torch.cat([cls2_old, cls2.unsqueeze(1)], dim = 1)
            distill_out_all = torch.cat([distillout_old, distill_out.unsqueeze(1)], dim = 1)
            cls_aux_all = torch.cat([cls_aux_old, cls_aux.unsqueeze(1)], dim = 1)
            exit_mask = selectors(cls1_all, cls2_all, cls_aux_all, distill_out_all, idx)

        exit_rate = torch.sum(exit_mask, dtype=torch.float32) / batch_size * 100.0
        acc_same, _ = accuracy(ens[exit_mask.type(torch.bool)], target[exit_mask.type(torch.bool)]) if target[exit_mask.type(torch.bool)].size(0) > 0 else (ens.new_tensor([0]), 0)
        acc_diff, _ = accuracy(ens[~exit_mask.type(torch.bool)], target[~exit_mask.type(torch.bool)]) if target[~exit_mask.type(torch.bool)].size(0) > 0 else (ens.new_tensor([0]), 0)
        acc_all, _ = accuracy(ens, target)

        Exit_rate.update(exit_rate.item(), batch_size)
        Aux_Head1_Acc1.update(acc_1[0].item(), batch_size)
        Aux_Head2_Acc1.update(acc_2[0].item(), batch_size)
        Acc_aux_ens.update(acc_aux_ens[0].item(), batch_size)
        Acc_ens.update(acc_ens[0].item(), batch_size)
        Acc_same.update(acc_same[0].item(), batch_size)
        Acc_diff.update(acc_diff[0].item(), batch_size)
        Acc_all.update(acc_all[0].item(), batch_size)
        Acc_ens_sf.update(acc_ens_sf[0].item(), batch_size)
        Batch_time.update(time.time() - end)
        
    logger.info(progress.display_avg())
    for k, v in progress.compose_json().items():
        writer.add_scalar(f"{k}/ind_evaluate_{split}_{idx+1}", v, epoch)
    writer.flush()

    return ens_sf, Acc_same.avg


@torch.no_grad()
def build_map_for_training(
        train_loader,
        estimator,
        idx,
        device,
        logger,
        args,
        selectors,
):

    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data_time', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [Batch_time, Data_time],
        prefix=f"Build Feat Map Ens:[{idx+1}]",
        logger=logger)
    end = time.time()
    for _, elem in enumerate(train_loader):
        
        Data_time.update(time.time() - end)
        data_id, data, _ = io.split_data_target(elem, device)

        cls1, cls2, distill_out = estimator(*data)
        cls_aux = (cls1 + cls2) / 2
        
        ens = F.softmax(cls_aux * (1 - args.alpha) + distill_out * args.alpha)
        _, pred_ens = ens.topk(1, 1, True, True)

        if idx == 0:
            mask_now = selectors(cls1.unsqueeze(1), cls2.unsqueeze(1), cls_aux.unsqueeze(1), distill_out.unsqueeze(1), idx)
        else:
            _, _, _, cls1_old, cls2_old, distillout_old, cls_aux_old = look_up(data_id)
            cls1_all = torch.cat([cls1_old, cls1.unsqueeze(1)], dim = 1)
            cls2_all = torch.cat([cls2_old, cls2.unsqueeze(1)], dim = 1)
            distill_out_all = torch.cat([distillout_old, distill_out.unsqueeze(1)], dim = 1)
            cls_aux_all = torch.cat([cls_aux_old, cls_aux.unsqueeze(1)], dim = 1)
            mask_now = selectors(cls1_all, cls2_all, cls_aux_all, distill_out_all, idx)

        aux_dis = torch.mean(dis_criterion(F.softmax(cls1, 1), F.softmax(cls2, 1)), dim=-1)
        for id, _dis, _mask, _ens, _cls1, _cls2, _distill_out, _cls_aux in zip(data_id, aux_dis, mask_now, ens, cls1, cls2, distill_out, cls_aux):
            look_up_map[id].append((_mask, _dis, _ens, _cls1, _cls2, _distill_out, _cls_aux))
        Batch_time.update(time.time() - end)
    logger.info(progress.display_avg())


@torch.no_grad()
def look_up(indexs):
    exit_mask, exit_dis, ens, cls1, cls2, distillout, cls_aux = [], [], [], [], [], [], []
    for id in indexs:
        ens_now = []
        exit_dis_now = []

        cls_1 = []
        cls_2 = []
        distill_out = []
        cls_aux_ = []

        for idx, (_exit_mask, _exit_distance, _ens, _cls1, _cls2, _distill_out, _cls_aux) in enumerate(look_up_map[id]):
            ens_now.append(_ens.view(1, -1))
            cls_1.append(_cls1.unsqueeze(0))
            cls_2.append(_cls2.unsqueeze(0))
            distill_out.append(_distill_out.unsqueeze(0))
            cls_aux_.append(_cls_aux.unsqueeze(0))
            exit_dis_now.append(_exit_distance)
            if idx == 0:
                exit_mask_now = _exit_mask
            else:
                exit_mask_now += _exit_mask

        exit_dis.append(torch.mean(torch.tensor(exit_dis_now)))
        exit_mask.append(~torch.tensor(exit_mask_now.type(torch.bool)))
        
        ens_now = torch.cat(ens_now)
        ens.append(ens_now)

        cls_1 = torch.cat(cls_1, dim = 0)
        cls1.append(cls_1)

        cls_2 = torch.cat(cls_2, dim = 0)
        cls2.append(cls_2)

        distill_out = torch.cat(distill_out, dim = 0)
        distillout.append(distill_out)

        cls_aux_ = torch.cat(cls_aux_, dim = 0)
        cls_aux.append(cls_aux_)

    ens = torch.stack(ens)
    exit_dis = _exit_distance.new_tensor(exit_dis)
    exit_mask = _exit_distance.new_tensor(exit_mask) == 1
    cls1 = torch.stack(cls1)
    cls2 = torch.stack(cls2)
    distillout = torch.stack(distillout)
    cls_aux = torch.stack(cls_aux)
    
    return exit_mask, exit_dis, ens, cls1, cls2, distillout, cls_aux


class EnsembleClassifier(BaseClassifier):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in EnsembleClassifier.""",
        "classifier_forward",
    )
    def __init__(
            self,
            estimator,
            n_estimators,
            estimator_args=None,
            cuda=True,
            n_jobs=None,
            logger=None,
            args=None,
            log_dir=None,
            selector=None
    ):
        super(EnsembleClassifier, self).__init__(estimator,
                                               n_estimators,
                                               estimator_args,
                                               cuda,
                                               n_jobs,
                                               logger,
                                               )
        self.args = args
        self.writer = SummaryWriter(log_dir)
        self.writer.flush()
        self.ens_eval_step = 0
        self.checkpoint_dir = Path(os.environ.get('AMLT_OUTPUT_DIR', '.'))
        self.estimators_ = nn.ModuleList()
        self.estimators = []
        self.selectors = selector.cuda()
        self.selector_optimizers = None
        self.selector_schedulers = None

    @torch.no_grad()
    def evaluate(self, test_loader, estimators_=None, save_dir=None, args=None, selectors=None, tag = 'ens_evaluate', margin = None):
        self.eval()
        if margin is not None:
            selectors.margin = torch.ones([1])*margin
        num_estimator = args.n_estimators
        
        EarlyExit_Acc1 = AverageMeter('EarlyExit_Acc1', ':6.2f')
        EarlyExit_sf_Acc1 = AverageMeter('EarlyExit_sf_Acc1', ':6.2f')
        AvgEns_Acc1 = AverageMeter('AvgEns_Acc1', ':6.2f')
        AvgEns_sf_Acc1 = AverageMeter('AvgEns_sf_Acc1', ':6.2f')
        Acc_same_l = [AverageMeter(f'Acc_same_{idx + 1}', ':6.2f') for idx in range(num_estimator)]
        Acc_diff_l = [AverageMeter(f'Acc_diff_{idx + 1}', ':6.2f') for idx in range(num_estimator)]
        Acc_all_l = [AverageMeter(f'Acc_all_{idx + 1}', ':6.2f') for idx in range(num_estimator)]
        Exit_rate_l = [AverageMeter(f'Exit_rate_{idx + 1}', ':6.2f') for idx in range(num_estimator)]
        Ind_Acc = [AverageMeter(f'Ind_Acc_{idx + 1}', ':6.2f') for idx in range(num_estimator)]
        progress = ProgressMeter(
            len(test_loader),
            [EarlyExit_Acc1, EarlyExit_sf_Acc1, AvgEns_Acc1, AvgEns_sf_Acc1] + \
            Exit_rate_l + Acc_same_l + Acc_diff_l + Acc_all_l + Ind_Acc,
            prefix=f"Eval: ",
            logger=self.logger)

        for _, elem in enumerate(test_loader):
            idx, data, target = io.split_data_target(
                elem, self.device
            )
            earlyexit_out = []
            earlyexit_out_sf = []
            avg_ens = []
            avg_ens_sf = []
            cls1_old = []
            cls2_old = []
            distillout_old = []
            cls_aux_old = []
            batch_size = target.size(0)

            exit_mask_before = torch.zeros(target.size(0)).cuda()

            for idx, estimator in enumerate(estimators_ if estimators_ else self.estimators_):
                
                cls1, cls2, distill_out = estimator(*data)
                cls_aux = (cls1 + cls2) / 2

                ens = F.softmax(cls_aux * (1 - args.alpha) + distill_out * args.alpha)
                ens_sf = F.softmax((F.softmax(cls1, 1) + F.softmax(cls2, 1)) * (1 - args.alpha)  + F.softmax(distill_out, 1) * args.alpha)

                avg_ens.append(ens.clone())
                avg_ens_sf.append(ens_sf.clone())

                cls1_old.append(cls1.unsqueeze(1))
                cls2_old.append(cls2.unsqueeze(1))
                distillout_old.append(distill_out.unsqueeze(1))
                cls_aux_old.append(cls_aux.unsqueeze(1))

                ind_acc_tmp, _ = accuracy(ens, target)
                Ind_Acc[idx].update(ind_acc_tmp[0].item(), batch_size)

                if idx > 0:
                    ens[exit_mask_before.type(torch.bool)] = earlyexit_out[-1][exit_mask_before.type(torch.bool)]
                    ens_sf[exit_mask_before.type(torch.bool)] = earlyexit_out_sf[-1][exit_mask_before.type(torch.bool)]

                earlyexit_out.append(ens)
                earlyexit_out_sf.append(ens_sf)

                if idx == 0:
                    mask_now = selectors(cls1.unsqueeze(1), cls2.unsqueeze(1), cls_aux.unsqueeze(1), distill_out.unsqueeze(1), idx)
                    exit_rate = torch.sum(mask_now, dtype=torch.float32) / data[0].size(0) * 100.0
                    acc_same, _ = accuracy(ens[mask_now.type(torch.bool)], target[mask_now.type(torch.bool)]) if target[mask_now.type(torch.bool)].size(0) > 0 else (ens.new_tensor([0]), 0)
                    acc_diff, _ = accuracy(ens[~mask_now.type(torch.bool)], target[~mask_now.type(torch.bool)]) if target[~mask_now.type(torch.bool)].size(0) > 0 else (ens.new_tensor([0]), 0)
                    acc_all, _ = accuracy(ens, target)
                else:
                    cls1_all = torch.cat(cls1_old, dim = 1)
                    cls2_all = torch.cat(cls2_old, dim = 1)
                    cls_aux_all = torch.cat(cls_aux_old, dim = 1)
                    distill_out_all = torch.cat(distillout_old, dim = 1)
                    mask_now = selectors(cls1_all, cls2_all, cls_aux_all, distill_out_all, idx)
                    mask_increase = (~exit_mask_before.type(torch.bool)).__and__(mask_now.type(torch.bool))
                    exit_rate = torch.sum(mask_increase, dtype=torch.float32) / data[0].size(0) * 100.0
                    acc_same, _ = accuracy(ens[mask_increase.type(torch.bool)], target[mask_increase.type(torch.bool)]) if target[mask_increase.type(torch.bool)].size(0) > 0 else (ens.new_tensor([0]), 0)
                    acc_diff, _ = accuracy(ens[~mask_increase.type(torch.bool)], target[~mask_increase.type(torch.bool)]) if target[~mask_increase.type(torch.bool)].size(0) > 0 else (ens.new_tensor([0]), 0)
                    acc_all, _ = accuracy(ens, target)

                exit_mask_before = (mask_now.type(torch.bool)).__or__(exit_mask_before.type(torch.bool))

                Exit_rate_l[idx].update(exit_rate.item(), batch_size)
                Acc_same_l[idx].update(acc_same[0].item(), batch_size)
                Acc_diff_l[idx].update(acc_diff[0].item(), batch_size)
                Acc_all_l[idx].update(acc_all[0].item(), batch_size)

            earlyexit_acc_1, _ = accuracy(op.average(earlyexit_out), target)
            earlyexit_sf_acc_1, _ = accuracy(op.average(earlyexit_out_sf), target)
            avg_ens_acc_1, _ = accuracy(op.average(avg_ens), target)
            avg_ens_sf_acc_1, _ = accuracy(op.average(avg_ens_sf), target)

            EarlyExit_Acc1.update(earlyexit_acc_1[0].item(), batch_size)
            EarlyExit_sf_Acc1.update(earlyexit_sf_acc_1[0].item(), batch_size)
            AvgEns_Acc1.update(avg_ens_acc_1[0].item(), batch_size)
            AvgEns_sf_Acc1.update(avg_ens_sf_acc_1[0].item(), batch_size)
            
        cost = 0
        exit_rate_all = 0
        for i in range(num_estimator):
            exit_rate_all += Exit_rate_l[i].avg * 0.01
            cost += Exit_rate_l[i].avg * 0.01 * (i + 1)
        cost += (1 - exit_rate_all) * num_estimator

        self.logger.info(progress.display_avg())
        self.logger.info(f'Now inference cost:{cost:.3f}x base model')

        self.writer.add_scalar(f"{'inference_cost'}/" + tag, cost, self.ens_eval_step)

        if save_dir is not None:
            res = progress.compose_json()
            res['inference cost'] = cost
            if margin is not None:
                res['test_margin'] = margin
                with open(f"{save_dir}/{margin}_res.json", "w") as f:
                    json.dump(res, f, indent=4, sort_keys=True)
            else:
                with open(f"{save_dir}/res.json", "w") as f:
                    json.dump(res, f, indent=4, sort_keys=True)

        for k, v in progress.compose_json().items():
            if estimators_ is not None:
                if k.split('_')[-1].isdigit() and int(k.split('_')[-1]) > len(estimators_):
                    continue
            self.writer.add_scalar(f"{k}/ens_evaluate", v, self.ens_eval_step)
        self.writer.flush()
        self.ens_eval_step += 1

        return EarlyExit_Acc1.avg

    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    def _make_selector(self):
        """Make and configure a copy of `self.base_estimator_`."""
        # Call `deepcopy` to make a base estimator
        selector = copy.deepcopy(self.base_selector_)
        return selector.to(self.device)

    def set_selector_optimizer(self, optimizer_name, **kwargs):
        """Set the parameter optimizer."""
        self.selector_optimizer_name = optimizer_name
        self.selector_optimizer_args = kwargs

    def _checkpoint(self, cur_epoch, train_idx):
        torch.save(
            {
                "model": self.state_dict(),
                "epoch": cur_epoch,
                "train_idx": train_idx,
            },
            self.checkpoint_dir / "resume.pth",
        )
        print(f"Checkpoint saved to {self.checkpoint_dir / 'resume.pth'}", __name__)

    def _scheduler_checkpoint(self, scheduler_):
        torch.save(
            {
                "scheduler_": scheduler_.state_dict(),
            },
            self.checkpoint_dir / "scheduler_.pth",
        )
        print(f"scheduler_ checkpoint saved to {self.checkpoint_dir / 'scheduler_.pth'}", __name__)

    def _scheduler_resume(self, scheduler_):
        if (self.checkpoint_dir / "scheduler_.pth").exists():
            print(f"Resume from {self.checkpoint_dir / 'scheduler_.pth'}", __name__)
            checkpoint = torch.load(self.checkpoint_dir / "scheduler_.pth")
            scheduler_.load_state_dict(checkpoint["scheduler_"])

            return scheduler_
        else:
            print(f"No checkpoint found in {self.checkpoint_dir}", __name__)
            return scheduler_

    def _resume(self):
        if (self.checkpoint_dir / "resume.pth").exists():
            print(f"Resume from {self.checkpoint_dir / 'resume.pth'}", __name__)
            checkpoint = torch.load(self.checkpoint_dir / "resume.pth")
            cur_epoch = checkpoint["epoch"]
            train_idx = checkpoint["train_idx"]
            self.load_state_dict(checkpoint["model"], False)
            
            for i in range(train_idx):
                io.load(self.estimators[i], self.save_dir, self.logger, i, self.optimizers[i], self.scheduler_)

            try:
                io.load(self.selectors, self.save_dir, self.logger, -2, self.selector_optimizers, self.selector_schedulers)
                io.load(self.estimators[train_idx], self.save_dir, self.logger, train_idx, self.optimizers[train_idx], self.scheduler_)
            except:
                pass

            return cur_epoch, train_idx
        
        else:
            print(f"No checkpoint found in {self.checkpoint_dir}", __name__)
            return 0, 0

    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        valid_loader=None,
        save_model=True,
        save_dir=None,
    ):

        start_epoch = 0
        start_index = 0
        self.save_dir = save_dir

        self.estimators = []
        for _ in range(self.n_estimators):
            self.estimators.append(self._make_estimator())
        self.estimators_dic = [[] for _ in range(self.n_estimators)]

        self.optimizers = []
        for i in range(self.n_estimators):
            self.optimizers.append(
                set_module.set_optimizer(
                    self.estimators[i], self.optimizer_name, **self.optimizer_args
                )
            )

        self.scheduler_ = set_module.set_scheduler(
                self.optimizers[start_index], self.scheduler_name, **self.scheduler_args
            )
        
        if start_epoch != 0:
            reinit_scheduler_flag = False
        else:
            reinit_scheduler_flag = True
            
        start_epoch, start_index = self._resume()

        if start_index > 0.0:
            for train_idx in range(start_index):
                build_map_for_training(train_loader, self.estimators[train_idx], train_idx, self.device,
                                    self.logger, self.args, self.selectors)

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Training loop
        for train_idx in range(start_index, self.n_estimators):

            self.selector_optimizers = torch.optim.Adam(self.selectors.parameters(), lr=self.args.selector_lr, eps=1e-8, weight_decay=self.args.selector_weight_decay)
            self.selector_schedulers = torch.optim.lr_scheduler.MultiStepLR(self.selector_optimizers, milestones=self.scheduler_args['milestones'], gamma=self.args.gamma)

            if self.use_scheduler_ and reinit_scheduler_flag:
                self.scheduler_ = set_module.set_scheduler(
                    self.optimizers[train_idx], self.scheduler_name, **self.scheduler_args
                )

            for epoch in range(start_epoch, epochs):
                self.train()
                if self.use_scheduler_:
                    cur_lr = self.scheduler_.get_last_lr()[0]
                    selector_lr = self.selector_schedulers.get_last_lr()[0]
                else:
                    cur_lr = None
                    selector_lr = None

                self.estimators[train_idx], self.optimizers[train_idx], self.selector_optimizers = _parallel_fit_per_epoch(
                    train_loader,
                    self.estimators,
                    cur_lr,
                    self.optimizers[train_idx],
                    self._criterion,
                    train_idx,
                    epoch,
                    log_interval,
                    self.device,
                    self.logger,
                    self.args,
                    self.writer,
                    self.selectors,
                    self.selector_optimizers,
                    selector_lr
                )

                with torch.no_grad():
                    self.eval()
                    if save_model and epoch % 5 == 0:
                        io.save(self.estimators[train_idx], save_dir, self.logger, train_idx, self.optimizers[train_idx], self.scheduler_)
                        self._checkpoint(epoch + 1 if epoch != epochs - 1 else 0, train_idx if epoch != epochs - 1 else train_idx + 1)
                        self._scheduler_checkpoint(self.scheduler_)
                        io.save(self.selectors, save_dir, self.logger, -2, self.selector_optimizers, self.selector_schedulers)

                    if test_loader:
                        _parallel_test_per_epoch(test_loader,
                                                 self.estimators[train_idx],
                                                 train_idx,
                                                 epoch,
                                                 self.device,
                                                 self.logger,
                                                 self.args,
                                                 self.writer,
                                                 split='test',
                                                 selectors=self.selectors,
                                                 )

                        if epoch % 5 == 0:
                            acc = self.evaluate(test_loader, estimators_= self.estimators[:train_idx + 1], args = self.args, selectors = self.selectors)
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(self.estimators)

                # Update the scheduler
                with warnings.catch_warnings():
                    # UserWarning raised by PyTorch is ignored because
                    # scheduler does not have a real effect on the optimizer.
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        self.scheduler_.step()
                        self.selector_schedulers.step()

            if valid_loader is None:
                build_map_for_training(train_loader, self.estimators[train_idx], train_idx, self.device,
                                    self.logger, self.args, self.selectors)
            else:
                self.estimators[train_idx].load_state_dict(self.estimators_dic[train_idx])
                build_map_for_training(train_loader, self.estimators[train_idx], train_idx, self.device,
                                    self.logger, self.args, self.selectors)
                build_map_for_training(valid_loader, self.estimators[train_idx], train_idx, self.device,
                                   self.logger, self.args, self.selectors)

            start_epoch = 0

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(self.estimators)
        self.selectors = self.selectors

        if save_model:
            io.save(self, save_dir, self.logger)