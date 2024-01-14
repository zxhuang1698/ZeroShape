import torch
import torch.nn as nn

from utils.util import EasyDict as edict
from utils.loss import Loss
from model.depth.dpt_depth import DPTDepthModel
from utils.layers import Bottleneck_Conv
from utils.camera import unproj_depth, valid_norm_fac

class Graph(nn.Module):

    def __init__(self, opt):
        super().__init__()
        # define the depth pred model based on omnidata
        self.dpt_depth = DPTDepthModel(backbone='vitb_rn50_384')
        if opt.arch.depth.pretrained is not None:
            checkpoint = torch.load(opt.arch.depth.pretrained, map_location="cuda:{}".format(opt.device))
            state_dict = checkpoint['model_state_dict']
            self.dpt_depth.load_state_dict(state_dict)
            
        if opt.loss_weight.intr is not None:
            self.intr_feat_channels = 768
            self.intr_head = nn.Sequential(
                Bottleneck_Conv(self.intr_feat_channels, kernel_size=3),
                Bottleneck_Conv(self.intr_feat_channels, kernel_size=3),
            )
            self.intr_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.intr_proj = nn.Linear(self.intr_feat_channels, 3)
            # init the last linear layer so it outputs zeros
            nn.init.zeros_(self.intr_proj.weight)
            nn.init.zeros_(self.intr_proj.bias)

        self.loss_fns = Loss(opt)

    def intr_param2mtx(self, opt, intr_params):
        '''
        Parameters:
            opt: config
            intr_params: [B, 3], [scale_f, delta_cx, delta_cy]
        Return:
            intr: [B, 3, 3]
        '''
        batch_size = len(intr_params)
        f = 1.3875
        intr = torch.zeros(3, 3).float().to(intr_params.device).unsqueeze(0).repeat(batch_size, 1, 1)
        intr[:, 2, 2] += 1
        # scale the focal length
        # range: [-1, 1], symmetric
        scale_f = torch.tanh(intr_params[:, 0])
        # range: [1/4, 4], symmetric
        scale_f = torch.pow(4. , scale_f)
        intr[:, 0, 0] += f * opt.W * scale_f
        intr[:, 1, 1] += f * opt.H * scale_f
        # shift the optic center, (at most to the image border)
        shift_cx = torch.tanh(intr_params[:, 1]) * opt.W / 2
        shift_cy = torch.tanh(intr_params[:, 2]) * opt.H / 2
        intr[:, 0, 2] += opt.W / 2 + shift_cx
        intr[:, 1, 2] += opt.H / 2 + shift_cy
        return intr

    def forward(self, opt, var, training=False, get_loss=True):
        batch_size = len(var.idx)

        # predict the depth map and feature maps if needed
        if opt.loss_weight.intr is None:
            var.depth_pred = self.dpt_depth(var.rgb_input_map)
        else:
            var.depth_pred, intr_feat = self.dpt_depth(var.rgb_input_map, get_feat=True)
            # predict the intrinsics
            intr_feat = self.intr_head(intr_feat)
            intr_feat = self.intr_pool(intr_feat).squeeze(-1).squeeze(-1)
            intr_params = self.intr_proj(intr_feat)
            # [B, 3, 3]
            var.intr_pred = self.intr_param2mtx(opt, intr_params)
            
            # project the predicted depth map to 3D points and normalize, [B, H*W, 3]
            seen_points_3D_pred = unproj_depth(opt, var.depth_pred, var.intr_pred)
            seen_points_mean_pred, seen_points_scale_pred = valid_norm_fac(seen_points_3D_pred, var.mask_input_map > 0.5)
            var.seen_points_pred = (seen_points_3D_pred - seen_points_mean_pred.unsqueeze(1)) / seen_points_scale_pred.unsqueeze(-1).unsqueeze(-1)
            var.seen_points_pred[(var.mask_input_map<=0.5).view(batch_size, -1)] = 0
            
            if 'depth_input_map' in var or training:
                # project the ground truth depth map to 3D points and normalize, [B, H*W, 3]
                seen_points_3D_gt = unproj_depth(opt, var.depth_input_map, var.intr)
                seen_points_mean_gt, seen_points_scale_gt = valid_norm_fac(seen_points_3D_gt, var.mask_input_map > 0.5)
                var.seen_points_gt = (seen_points_3D_gt - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
                var.seen_points_gt[(var.mask_input_map<=0.5).view(batch_size, -1)] = 0
                
                # record the validity mask, [B, H*W]
                var.validity_mask = (var.mask_input_map>0.5).float().view(batch_size, -1)
            
        # calculate the loss if needed
        if get_loss: 
            loss = self.compute_loss(opt, var, training)
            return var, loss
        
        return var

    def compute_loss(self, opt, var, training=False):
        loss = edict()
        if opt.loss_weight.depth is not None:
            loss.depth = self.loss_fns.depth_loss(var.depth_pred, var.depth_input_map, var.mask_input_map)
        if opt.loss_weight.intr is not None:
            loss.intr = self.loss_fns.intr_loss(var.seen_points_pred, var.seen_points_gt, var.validity_mask)
        return loss

