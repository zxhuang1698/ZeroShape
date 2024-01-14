import torch
import torch.nn as nn

from utils.util import EasyDict as edict
from utils.loss import Loss
from model.shape.implicit import Implicit
from model.shape.seen_coord_enc import CoordEncAtt, CoordEncRes
from model.shape.rgb_enc import RGBEncAtt, RGBEncRes
from model.depth.dpt_depth import DPTDepthModel
from utils.util import toggle_grad, interpolate_coordmap, get_child_state_dict
from utils.camera import unproj_depth, valid_norm_fac
from utils.layers import Bottleneck_Conv

class Graph(nn.Module):

    def __init__(self, opt):
        super().__init__()
        # define the intrinsics head
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
            
        # define the depth pred model based on omnidata
        self.dpt_depth = DPTDepthModel(backbone='vitb_rn50_384')
        # load the pretrained depth model
        # when intrinsics need to be predicted we need to load that part as well
        self.load_pretrained_depth(opt)
        if opt.optim.fix_dpt:
            toggle_grad(self.dpt_depth, False)
            toggle_grad(self.intr_head, False)
            toggle_grad(self.intr_proj, False)

        # encoder that encode seen surface to impl conditioning vec
        if opt.arch.depth.encoder == 'resnet':
            opt.arch.depth.dsp = 1
            self.coord_encoder = CoordEncRes(opt)
        else:
            self.coord_encoder = CoordEncAtt(embed_dim=opt.arch.latent_dim, n_blocks=opt.arch.depth.n_blocks, 
                                        num_heads=opt.arch.num_heads, win_size=opt.arch.win_size//opt.arch.depth.dsp)
            
        # rgb branch (not used in final model, keep here for extension)
        if opt.arch.rgb.encoder == 'resnet':
            self.rgb_encoder = RGBEncRes(opt)
        elif opt.arch.rgb.encoder == 'transformer':
            self.rgb_encoder = RGBEncAtt(img_size=opt.H, embed_dim=opt.arch.latent_dim, n_blocks=opt.arch.rgb.n_blocks, 
                                        num_heads=opt.arch.num_heads, win_size=opt.arch.win_size)
        else:
            self.rgb_encoder = None
        
        # implicit function
        feat_res = opt.H // opt.arch.win_size
        self.impl_network = Implicit(feat_res**2, latent_dim=opt.arch.latent_dim*2 if self.rgb_encoder else opt.arch.latent_dim, 
                                     semantic=self.rgb_encoder is not None, n_channels=opt.arch.impl.n_channels, 
                                     n_blocks_attn=opt.arch.impl.att_blocks, n_layers_mlp=opt.arch.impl.mlp_layers, 
                                     num_heads=opt.arch.num_heads, posenc_3D=opt.arch.impl.posenc_3D, 
                                     mlp_ratio=opt.arch.impl.mlp_ratio, skip_in=opt.arch.impl.skip_in, 
                                     pos_perlayer=opt.arch.impl.posenc_perlayer)
        
        # loss functions
        self.loss_fns = Loss(opt)
            
    def load_pretrained_depth(self, opt):
        if opt.pretrain.depth:
            # loading from our pretrained depth and intr model
            if opt.device == 0:
                print("loading dpt depth from {}...".format(opt.pretrain.depth))
            checkpoint = torch.load(opt.pretrain.depth, map_location="cuda:{}".format(opt.device))
            self.dpt_depth.load_state_dict(get_child_state_dict(checkpoint["graph"], "dpt_depth"))
            # load the intr head
            if opt.device == 0:
                print("loading pretrained intr from {}...".format(opt.pretrain.depth))
            self.intr_head.load_state_dict(get_child_state_dict(checkpoint["graph"], "intr_head"))
            self.intr_proj.load_state_dict(get_child_state_dict(checkpoint["graph"], "intr_proj"))
        elif opt.arch.depth.pretrained:
            # loading from omnidata weights
            if opt.device == 0:
                print("loading dpt depth from {}...".format(opt.arch.depth.pretrained))
            checkpoint = torch.load(opt.arch.depth.pretrained, map_location="cuda:{}".format(opt.device))
            state_dict = checkpoint['model_state_dict']
            self.dpt_depth.load_state_dict(state_dict)

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
        
        # encode the rgb, [B, 3, H, W] -> [B, 1+H/(ws)*W/(ws), C], not used in our final model
        var.latent_semantic = self.rgb_encoder(var.rgb_input_map) if self.rgb_encoder else None

        # predict the depth map and intrinsics
        var.depth_pred, intr_feat = self.dpt_depth(var.rgb_input_map, get_feat=True)
        depth_map = var.depth_pred
        # predict the intrinsics
        intr_feat = self.intr_head(intr_feat)
        intr_feat = self.intr_pool(intr_feat).squeeze(-1).squeeze(-1)
        intr_params = self.intr_proj(intr_feat)
        # [B, 3, 3]
        var.intr_pred = self.intr_param2mtx(opt, intr_params)
        intr_forward = var.intr_pred
        # record the validity mask, [B, H*W]
        var.validity_mask = (var.mask_input_map>0.5).float().view(batch_size, -1)

        # project the depth to 3D points in view-centric frame
        # [B, H*W, 3], in camera coordinates
        seen_points_3D_pred = unproj_depth(opt, depth_map, intr_forward)
        # [B, H*W, 3], [B, 1, H, W] (boolean) -> [B, 3], [B]
        seen_points_mean_pred, seen_points_scale_pred = valid_norm_fac(seen_points_3D_pred, var.mask_input_map > 0.5)
        # normalize the seen surface, [B, H*W, 3]
        var.seen_points = (seen_points_3D_pred - seen_points_mean_pred.unsqueeze(1)) / seen_points_scale_pred.unsqueeze(-1).unsqueeze(-1)
        var.seen_points[(var.mask_input_map<=0.5).view(batch_size, -1)] = 0
        # [B, 3, H, W]
        seen_3D_map = var.seen_points.view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2).contiguous()
        seen_3D_dsp, mask_dsp = interpolate_coordmap(seen_3D_map, var.mask_input_map, (opt.H//opt.arch.depth.dsp, opt.W//opt.arch.depth.dsp))
        
        # encode the depth, [B, 1, H/k, W/k] -> [B, 1+H/(ws)*W/(ws), C]
        if opt.arch.depth.encoder == 'resnet':
            var.latent_depth = self.coord_encoder(seen_3D_dsp, mask_dsp)
        else:
            var.latent_depth = self.coord_encoder(seen_3D_dsp.permute(0, 2, 3, 1).contiguous(), mask_dsp.squeeze(1)>0.5)
        

        var.pose = var.pose_gt
        # forward for loss calculation (only during training)
        if 'gt_sample_points' in var and 'gt_sample_sdf' in var:
            with torch.no_grad():
                # get the normalizing fac based on the GT seen surface
                # project the GT depth to 3D points in view-centric frame
                # [B, H*W, 3], in camera coordinates
                seen_points_3D_gt = unproj_depth(opt, var.depth_input_map, var.intr)
                # [B, H*W, 3], [B, 1, H, W] (boolean) -> [B, 3], [B]
                seen_points_mean_gt, seen_points_scale_gt = valid_norm_fac(seen_points_3D_gt, var.mask_input_map > 0.5)
                var.seen_points_gt = (seen_points_3D_gt - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
                var.seen_points_gt[(var.mask_input_map<=0.5).view(batch_size, -1)] = 0
                
                # transform the GT points accordingly
                # [B, 3, 3]
                R_gt = var.pose_gt[:, :, :3]
                # [B, 3, 1]
                T_gt = var.pose_gt[:, :, 3:]
                # [B, 3, N]
                gt_sample_points_transposed = var.gt_sample_points.permute(0, 2, 1).contiguous()
                # camera coordinates, [B, N, 3]
                gt_sample_points_cam = (R_gt @ gt_sample_points_transposed + T_gt).permute(0, 2, 1).contiguous()
                # normalize with seen std and mean, [B, N, 3]
                var.gt_points_cam = (gt_sample_points_cam - seen_points_mean_gt.unsqueeze(1)) / seen_points_scale_gt.unsqueeze(-1).unsqueeze(-1)
                
                # get near-surface points for visualization
                # [B, 100, 3]
                close_surf_idx = torch.topk(var.gt_sample_sdf.abs(), k=100, dim=1, largest=False)[1].unsqueeze(-1).repeat(1, 1, 3)
                # [B, 100, 3]
                var.gt_surf_points = torch.gather(var.gt_points_cam, dim=1, index=close_surf_idx)
        
            # [B, N], [B, N, 1+feat_res**2], inference the impl_network for 3D loss
            var.pred_sample_occ, attn = self.impl_network(var.latent_depth, var.latent_semantic, var.gt_points_cam)

        # calculate the loss if needed
        if get_loss: 
            loss = self.compute_loss(opt, var, training)
            return var, loss
        
        return var

    def compute_loss(self, opt, var, training=False):
        loss = edict()
        if opt.loss_weight.depth is not None:
            loss.depth = self.loss_fns.depth_loss(var.depth_pred, var.depth_input_map, var.mask_input_map)
        if opt.loss_weight.intr is not None and training:
            loss.intr = self.loss_fns.intr_loss(var.seen_points, var.seen_points_gt, var.validity_mask)
        if opt.loss_weight.shape is not None and training:
            loss.shape = self.loss_fns.shape_loss(var.pred_sample_occ, var.gt_sample_sdf)
        return loss
