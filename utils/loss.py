import torch
import torch.nn as nn
import torch.nn.functional as torch_F

from copy import deepcopy
from model.depth.midas_loss import MidasLoss

class Loss(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = deepcopy(opt)
        self.occ_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.midas_loss = MidasLoss(alpha=opt.training.depth_loss.grad_reg, 
                                    inverse_depth=opt.training.depth_loss.depth_inv, 
                                    shrink_mask=opt.training.depth_loss.mask_shrink)

    def shape_loss(self, pred_occ_raw, gt_sdf):
        assert len(pred_occ_raw.shape) == 2
        assert len(gt_sdf.shape) == 2
        # [B, N]
        gt_occ = (gt_sdf < 0).float()
        loss = self.occ_loss(pred_occ_raw, gt_occ)
        weight_mask = torch.ones_like(loss)
        thres = self.opt.training.shape_loss.impt_thres
        weight_mask[torch.abs(gt_sdf) < thres] = weight_mask[torch.abs(gt_sdf) < thres] * self.opt.training.shape_loss.impt_weight 
        loss = loss * weight_mask
        return loss.mean()

    def depth_loss(self, pred_depth, gt_depth, mask):
        assert len(pred_depth.shape) == len(gt_depth.shape) == len(mask.shape) == 4
        assert pred_depth.shape[1] == gt_depth.shape[1] == mask.shape[1] == 1
        loss = self.midas_loss(pred_depth, gt_depth, mask)
        return loss
    
    def intr_loss(self, seen_pred, seen_gt, mask):
        assert len(seen_pred.shape) == len(seen_gt.shape) == 3
        assert len(mask.shape) == 2
        # [B, HW]
        distance = torch.sum((seen_pred - seen_gt)**2, dim=-1)
        loss = (distance * mask).sum() / (mask.sum() + 1.e-8)
        return loss