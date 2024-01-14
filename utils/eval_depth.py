# based on https://gist.github.com/ranftlr/45f4c7ddeb1bbb88d606bc600cab6c8d

import torch

class DepthMetric:
    def __init__(self, thresholds=[1.25, 1.25**2, 1.25**3], depth_cap=None, prediction_type='depth'):
        self.thresholds = thresholds
        self.depth_cap = depth_cap
        self.metric_keys = self.get_metric_keys()
        self.prediction_type = prediction_type

    def compute_scale_and_shift(self, prediction, target, mask):
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
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def get_metric_keys(self):
        metric_keys = []
        for threshold in self.thresholds:
            metric_keys.append('d>{}'.format(threshold))
        metric_keys.append('rmse')
        metric_keys.append('l1_err')
        metric_keys.append('abs_rel')
        return metric_keys

    def compute_metrics(self, prediction, target, mask):
        # check inputs
        prediction = prediction.float()
        target = target.float()
        mask = mask.float()
        assert prediction.shape == target.shape == mask.shape
        assert len(prediction.shape) == 4
        assert prediction.shape[1] == 1
        assert prediction.dtype == target.dtype == mask.dtype == torch.float32
        
        # process inputs
        prediction = prediction.squeeze(1)
        target = target.squeeze(1)
        mask = (mask.squeeze(1) > 0.5).long()
        
        # output dict
        metrics = {}
        
        # get the predicted disparity
        prediction_disparity = torch.zeros_like(prediction)
        if self.prediction_type == 'depth':
            prediction_disparity[mask == 1] = 1.0 / (prediction[mask == 1] + 1.e-6)
        elif self.prediction_type == 'disparity':
            prediction_disparity[mask == 1] = prediction[mask == 1]
        else:
            raise ValueError('Unknown prediction type: {}'.format(self.prediction_type))
        
        # transform predicted disparity to align with depth
        target_disparity = torch.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / target[mask == 1]
        scale, shift = self.compute_scale_and_shift(prediction_disparity, target_disparity, mask)
        prediction_aligned = scale.view(-1, 1, 1) * prediction_disparity + shift.view(-1, 1, 1)

        if self.depth_cap is not None:
            disparity_cap = 1.0 / self.depth_cap
            prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediciton_depth = 1.0 / prediction_aligned

        # delta > threshold, [batch_size, ]
        for threshold in self.thresholds:
            err = torch.zeros_like(prediciton_depth, dtype=torch.float)
            err[mask == 1] = torch.max(
                prediciton_depth[mask == 1] / target[mask == 1],
                target[mask == 1] / prediciton_depth[mask == 1],
            )
            err[mask == 1] = (err[mask == 1] > threshold).float()
            metrics['d>{}'.format(threshold)] = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))
        
        # rmse, [batch_size, ]
        rmse = torch.zeros_like(prediciton_depth, dtype=torch.float)
        rmse[mask == 1] = (prediciton_depth[mask == 1] - target[mask == 1]) ** 2
        rmse = torch.sum(rmse, (1, 2)) / torch.sum(mask, (1, 2))
        metrics['rmse'] = torch.sqrt(rmse)
        
        # l1 error, [batch_size, ]
        l1_err = torch.zeros_like(prediciton_depth, dtype=torch.float)
        l1_err[mask == 1] = torch.abs(prediciton_depth[mask == 1] - target[mask == 1])
        metrics['l1_err'] = torch.sum(l1_err, (1, 2)) / torch.sum(mask, (1, 2))
        
        # abs_rel, [batch_size, ]
        abs_rel = torch.zeros_like(prediciton_depth, dtype=torch.float)
        abs_rel[mask == 1] = torch.abs(prediciton_depth[mask == 1] - target[mask == 1]) / target[mask == 1]
        metrics['abs_rel'] = torch.sum(abs_rel, (1, 2)) / torch.sum(mask, (1, 2))

        return metrics, prediciton_depth.unsqueeze(1)

