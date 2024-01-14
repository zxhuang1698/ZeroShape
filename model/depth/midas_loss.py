# modified from https://github.com/EPFL-VILAB/omnidata
import torch
import torch.nn as nn
import numpy as np

def masked_l1_loss(preds, target, mask_valid):
    element_wise_loss = abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    return element_wise_loss.sum() / (mask_valid.sum() + 1.e-6)

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / (det[valid] + 1e-6)
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / (det[valid] + 1e-6)

    return x_0, x_1


def masked_shift_and_scale(depth_preds, depth_gt, mask_valid):
    depth_preds_nan = depth_preds.clone()
    depth_gt_nan = depth_gt.clone()
    depth_preds_nan[~mask_valid] = np.nan
    depth_gt_nan[~mask_valid] = np.nan

    mask_diff = mask_valid.view(mask_valid.size()[:2] + (-1,)).sum(-1, keepdims=True) + 1

    # flatten spatial dimension and take valid median [B, 1, 1, 1]
    t_gt = depth_gt_nan.view(depth_gt_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_gt[torch.isnan(t_gt)] = 0
    # subtract median and set invalid position to 0
    diff_gt = torch.abs(depth_gt - t_gt)
    diff_gt[~mask_valid] = 0
    # get the avg abs diff value over valid regions [B, 1, 1, 1]
    s_gt = (diff_gt.view(diff_gt.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    # normalize
    depth_gt_aligned = (depth_gt - t_gt) / (s_gt + 1e-6)

    # same as gt normalization
    t_pred = depth_preds_nan.view(depth_preds_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_pred[torch.isnan(t_pred)] = 0
    diff_pred = torch.abs(depth_preds - t_pred)
    diff_pred[~mask_valid] = 0
    s_pred = (diff_pred.view(diff_pred.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_pred_aligned = (depth_preds - t_pred) / (s_pred + 1e-6)

    return depth_pred_aligned, depth_gt_aligned


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)



def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)



class SSIMAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth_preds, depth_gt, mask_valid):
        depth_pred_aligned, depth_gt_aligned = masked_shift_and_scale(depth_preds, depth_gt, mask_valid)
        ssi_mae_loss = masked_l1_loss(depth_pred_aligned, depth_gt_aligned, mask_valid)
        return ssi_mae_loss


class GradientMatchingTerm(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class MidasLoss(nn.Module):
    def __init__(self, alpha=0.1, scales=4, reduction='image-based', inverse_depth=True, shrink_mask=False):
        super().__init__()

        self.__ssi_mae_loss = SSIMAE()
        self.__gradient_matching_term = GradientMatchingTerm(scales=scales, reduction=reduction)
        self.__alpha = alpha
        self.inverse_depth = inverse_depth
        self.shrink_mask = shrink_mask

    # decrease valid region via min-pooling
    @torch.no_grad()
    def erode_mask(self, mask, max_pool_size=4):
        mask_float = mask.float()
        h, w = mask_float.shape[2], mask_float.shape[3]
        mask_float = 1 - mask_float
        mask_float = torch.nn.functional.max_pool2d(mask_float, kernel_size=max_pool_size)
        mask_float = torch.nn.functional.interpolate(mask_float, (h, w), mode='nearest')
        # only if a 4x4 region is all valid then we make that valid
        mask_valid = mask_float == 0
        return mask_valid

    def forward(self, prediction_raw, target_raw, mask_raw):
        if self.shrink_mask:
            mask = self.erode_mask(mask_raw)
        else:
            mask = mask_raw > 0.5
        ssi_loss = self.__ssi_mae_loss(prediction_raw, target_raw, mask)
        if self.__alpha <= 0:
            return ssi_loss
        
        if self.inverse_depth:
            prediction = 1 / (prediction_raw.squeeze(1) + 1e-6)
            target = 1 / (target_raw.squeeze(1) + 1e-6)
        else:
            prediction = prediction_raw.squeeze(1)
            target = target_raw.squeeze(1)
        # gradient loss
        scale, shift = compute_scale_and_shift(prediction, target, mask.squeeze(1))
        prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        reg_loss = self.__gradient_matching_term(prediction_ssi, target, mask.squeeze(1))
        total = ssi_loss + self.__alpha * reg_loss

        return total
