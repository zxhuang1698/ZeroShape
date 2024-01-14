import numpy as np
import os, time, datetime
import torch
import torch.utils.tensorboard
import importlib
import shutil
import utils.util as util
import utils.util_vis as util_vis

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.util import print_eval, setup, cleanup
from utils.util import EasyDict as edict
from utils.eval_depth import DepthMetric
from copy import deepcopy
from model.compute_graph import graph_depth

# ============================ main engine for training and evaluation ============================

class Runner():

    def __init__(self, opt):
        super().__init__()
        if os.path.isdir(opt.output_path) and opt.resume == False and opt.device == 0:
            for filename in os.listdir(opt.output_path):
                if "tfevents" in filename: os.remove(os.path.join(opt.output_path, filename))
                if "html" in filename: os.remove(os.path.join(opt.output_path, filename))
                if "vis" in filename: shutil.rmtree(os.path.join(opt.output_path, filename))
                if "dump" in filename: shutil.rmtree(os.path.join(opt.output_path, filename))
                if "embedding" in filename: shutil.rmtree(os.path.join(opt.output_path, filename))
        if opt.device == 0: 
            os.makedirs(opt.output_path,exist_ok=True)
        setup(opt.device, opt.world_size, opt.port)
        opt.batch_size = opt.batch_size // opt.world_size
    
    def get_viz_data(self, opt):
        # get data for visualization
        viz_data_list = []
        sample_range = len(self.viz_loader)
        viz_interval = sample_range // opt.eval.n_vis
        for i in range(sample_range):
            current_batch = next(self.viz_loader_iter)
            if i % viz_interval != 0: continue
            viz_data_list.append(current_batch)
        return viz_data_list
    
    def load_dataset(self, opt, eval_split="test"):
        data_train = importlib.import_module('data.{}'.format(opt.data.dataset_train))
        data_test = importlib.import_module('data.{}'.format(opt.data.dataset_test))
        if opt.device == 0: print("loading training data...")
        self.batch_order = []
        self.train_data = data_train.Dataset(opt, split="train", load_3D=False)
        self.train_loader = self.train_data.setup_loader(opt, shuffle=True, use_ddp=True, drop_last=True)
        self.num_batches = len(self.train_loader)
        if opt.device == 0: print("loading test data...")
        self.test_data = data_test.Dataset(opt, split=eval_split, load_3D=False)
        self.test_loader = self.test_data.setup_loader(opt, shuffle=False, use_ddp=True, drop_last=True, batch_size=opt.eval.batch_size)
        self.num_batches_test = len(self.test_loader)
        if len(self.test_loader.sampler) * opt.world_size < len(self.test_data):
            self.aux_test_dataset = torch.utils.data.Subset(self.test_data,
                                    range(len(self.test_loader.sampler) * opt.world_size, len(self.test_data)))
            self.aux_test_loader = torch.utils.data.DataLoader(
                self.aux_test_dataset, batch_size=opt.eval.batch_size, shuffle=False, drop_last=False,
                num_workers=opt.data.num_workers)
        if opt.device == 0:
            print("creating data for visualization...")
            self.viz_loader = self.test_data.setup_loader(opt, shuffle=False, use_ddp=False, drop_last=False, batch_size=1)
            self.viz_loader_iter = iter(self.viz_loader)
            self.viz_data = self.get_viz_data(opt)

    def build_networks(self, opt):
        if opt.device == 0: print("building networks...")
        self.graph = DDP(graph_depth.Graph(opt).to(opt.device), device_ids=[opt.device], find_unused_parameters=True)
        self.depth_metric = DepthMetric(thresholds=opt.eval.d_thresholds, depth_cap=opt.eval.depth_cap)

# =================================================== set up training =========================================================

    def setup_optimizer(self, opt):
        if opt.device == 0: print("setting up optimizers...")
        param_nodecay = []
        param_decay = []
        for name, param in self.graph.named_parameters():
            # skip and fixed params
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                # print("{} -> finetune_param_nodecay".format(name))
                param_nodecay.append(param)
            else:
                param_decay.append(param)
                # print("{} -> finetune_param_decay".format(name))
            # create the optim dictionary
            optim_dict = [
                {'params': param_nodecay, 'lr': opt.optim.lr, 'weight_decay': 0.},
                {'params': param_decay, 'lr': opt.optim.lr, 'weight_decay': opt.optim.weight_decay}
            ]
            
        self.optim = torch.optim.AdamW(optim_dict, betas=(0.9, 0.95))
        if opt.optim.sched: 
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, opt.max_epoch)
        if opt.optim.amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def restore_checkpoint(self, opt, best=False, evaluate=False):
        epoch_start, iter_start = None, None
        if opt.resume:
            if opt.device == 0: print("resuming from previous checkpoint...")
            epoch_start, iter_start, best_val, best_ep = util.restore_checkpoint(opt, self, resume=opt.resume, best=best, evaluate=evaluate)
            self.best_val = best_val
            self.best_ep = best_ep
        elif opt.load is not None:
            if opt.device == 0: print("loading weights from checkpoint {}...".format(opt.load))
            epoch_start, iter_start, best_val, best_ep = util.restore_checkpoint(opt, self, load_name=opt.load)
        else:
            if opt.device == 0: print("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def setup_visualizer(self, opt, test=False):
        if opt.device == 0: 
            print("setting up visualizers...")
            if opt.tb:
                self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path, flush_secs=10)
    
    def train(self, opt):
        # before training
        torch.cuda.set_device(opt.device)
        torch.cuda.empty_cache()
        if opt.device == 0: print("TRAINING START")
        self.train_metric_logger = util.MetricLogger(delimiter="  ")
        self.train_metric_logger.add_meter('lr', util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.iter_skip = self.iter_start % len(self.train_loader)
        self.it = self.iter_start
        self.skip_dis = False
        if not opt.resume: 
            self.best_val = np.inf
            self.best_ep = 1
        # training
        if self.iter_start == 0 and not opt.debug: self.evaluate(opt, ep=0, training=True)
        # if opt.device == 0: self.save_checkpoint(opt, ep=0, it=0, best_val=self.best_val, best_ep=self.best_ep)
        self.ep = self.epoch_start
        for self.ep in range(self.epoch_start, opt.max_epoch):
            self.train_epoch(opt)
        # after training
        if opt.device == 0: self.save_checkpoint(opt, ep=self.ep, it=self.it, best_val=self.best_val, best_ep=self.best_ep)
        if opt.tb and opt.device == 0:
            self.tb.flush()
            self.tb.close()
        if opt.device == 0: 
            print("TRAINING DONE")
            print("Best val: %.4f @ epoch %d" % (self.best_val, self.best_ep))
        cleanup()
    
    def train_epoch(self, opt):
        # before train epoch
        self.train_loader.sampler.set_epoch(self.ep)
        if opt.device == 0:
            print("training epoch {}".format(self.ep+1))
        batch_progress = range(self.num_batches)
        self.graph.train()
        # train epoch
        loader = iter(self.train_loader)
           
        for batch_id in batch_progress:
            # if resuming from previous checkpoint, skip until the last iteration number is reached
            if self.iter_skip>0:
                self.iter_skip -= 1
                continue
            batch = next(loader)
            # train iteration
            var = edict(batch)
            opt.H, opt.W = opt.image_size
            var = util.move_to_device(var, opt.device)
            loss = self.train_iteration(opt, var, batch_progress)
            
        # after train epoch
        lr = self.sched.get_last_lr()[0] if opt.optim.sched else opt.optim.lr
        if opt.optim.sched: self.sched.step()
        if (self.ep + 1) % opt.freq.eval == 0:
            if opt.device == 0: print("validating epoch {}".format(self.ep+1))
            current_val = self.evaluate(opt, ep=self.ep+1, training=True)
            if current_val < self.best_val and opt.device == 0:
                self.best_val = current_val
                self.best_ep = self.ep + 1
                self.save_checkpoint(opt, ep=self.ep, it=self.it, best_val=self.best_val, best_ep=self.best_ep, best=True, latest=True)

    def train_iteration(self, opt, var, loader):
        # before train iteration
        torch.distributed.barrier()
        
        # train iteration
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt.optim.amp):
            var, loss = self.graph.forward(opt, var, training=True, get_loss=True)
            loss = self.summarize_loss(opt, var, loss)
            loss_scaled = loss.all / opt.optim.accum
            
        # backward
        if opt.optim.amp:
            self.scaler.scale(loss_scaled).backward()
            # skip update if accumulating gradient
            if (self.it + 1) % opt.optim.accum == 0:
                self.scaler.unscale_(self.optim)
                # gradient clipping
                if opt.optim.clip_norm:
                    norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), opt.optim.clip_norm)
                    if opt.debug: print("Grad norm: {}".format(norm))
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
        else:
            loss_scaled.backward()
            if (self.it + 1) % opt.optim.accum == 0:
                if opt.optim.clip_norm:
                    norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), opt.optim.clip_norm)
                    if opt.debug: print("Grad norm: {}".format(norm))
                self.optim.step()
                self.optim.zero_grad()

        # after train iteration
        lr = self.sched.get_last_lr()[0] if opt.optim.sched else opt.optim.lr
        self.train_metric_logger.update(lr=lr)
        self.train_metric_logger.update(loss=loss.all)
        if opt.device == 0: 
            if (self.it) % opt.freq.vis == 0 and not opt.debug: 
                self.visualize(opt, var, step=self.it, split="train")
            if (self.it+1) % opt.freq.ckpt_latest == 0 and not opt.debug: 
                self.save_checkpoint(opt, ep=self.ep, it=self.it+1, best_val=self.best_val, best_ep=self.best_ep, latest=True)
            if (self.it) % opt.freq.scalar == 0 and not opt.debug: 
                self.log_scalars(opt, var, loss, step=self.it, split="train")
            if (self.it) % (opt.freq.save_vis * (self.it//10000*10+1)) == 0 and not opt.debug: 
                self.vis_train_iter(opt)
            if (self.it) % opt.freq.print == 0:
                print('[{}] '.format(datetime.datetime.now().time()), end='')
                print(f'Train Iter {self.it}/{self.num_batches*opt.max_epoch}: {self.train_metric_logger}')
        self.it += 1
        return loss
        
    @torch.no_grad()
    def vis_train_iter(self, opt):
        self.graph.eval()
        for i in range(len(self.viz_data)):
            var_viz = edict(deepcopy(self.viz_data[i]))
            var_viz = util.move_to_device(var_viz, opt.device)
            var_viz = self.graph.module(opt, var_viz, training=False, get_loss=False)
            vis_folder = "vis_log/iter_{}".format(self.it)
            os.makedirs("{}/{}".format(opt.output_path, vis_folder), exist_ok=True)
            util_vis.dump_images(opt, var_viz.idx, "image_input", var_viz.rgb_input_map, masks=None, from_range=(0, 1), folder=vis_folder)
            util_vis.dump_images(opt, var_viz.idx, "mask_input", var_viz.mask_input_map, folder=vis_folder)
            util_vis.dump_depths(opt, var_viz.idx, "depth_est", var_viz.depth_pred, var_viz.mask_input_map, rescale=True, folder=vis_folder)
            util_vis.dump_depths(opt, var_viz.idx, "depth_input", var_viz.depth_input_map, var_viz.mask_input_map, rescale=True, folder=vis_folder)
            if 'seen_points_pred' in var_viz and 'seen_points_gt' in var_viz:
                util_vis.dump_pointclouds_compare(opt, var_viz.idx, "seen_surface", var_viz.seen_points_pred, var_viz.seen_points_gt, folder=vis_folder)
        self.graph.train()

    def summarize_loss(self, opt, var, loss, non_act_loss_key=[]):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight)
            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key].mean()), "loss {} is Inf".format(key)
                assert not torch.isnan(loss[key].mean()), "loss {} is NaN".format(key)
                loss_all += float(opt.loss_weight[key])*loss[key].mean() if key not in non_act_loss_key else 0.0*loss[key].mean()
        loss.update(all=loss_all)
        return loss

# =================================================== set up evaluation =========================================================

    @torch.no_grad()
    def evaluate(self, opt, ep, training=False):
        self.graph.eval()
        loss_eval = edict()
        
        # metric dictionary
        metric_eval = {}
        for metric_key in self.depth_metric.metric_keys:
            metric_eval[metric_key] = []
        metric_avg = {}
        eval_metric_logger = util.MetricLogger(delimiter="  ")
        
        # dataloader on the test set
        with torch.cuda.device(opt.device):
            for it, batch in enumerate(self.test_loader):
                
                # inference the model
                var = edict(batch)
                var = self.evaluate_batch(opt, var, ep, it, single_gpu=False)

                # record foreground mae for evaluation
                sample_metrics, var.depth_pred_aligned = self.depth_metric.compute_metrics(
                    var.depth_pred, var.depth_input_map, var.mask_eroded if 'mask_eroded' in var else var.mask_input_map)
                var.rmse = sample_metrics['rmse']
                curr_metrics = {}
                for metric_key in metric_eval:
                    metric_eval[metric_key].append(sample_metrics[metric_key])
                    curr_metrics[metric_key] = sample_metrics[metric_key].mean()
                eval_metric_logger.update(**curr_metrics)
                # eval_metric_logger.update(metric_key=sample_metrics[metric_key].mean())
                
                # accumulate the scores
                if opt.device == 0 and it % opt.freq.print_eval == 0: 
                    print('[{}] '.format(datetime.datetime.now().time()), end='')
                    print(f'Eval Iter {it}/{len(self.test_loader)} @ EP {ep}: {eval_metric_logger}')
                
                # dump the result if in eval mode
                if not training: 
                    self.dump_results(opt, var, ep, write_new=(it == 0))

                # save the visualization
                if it == 0 and training and opt.device == 0: 
                    print("visualizing and saving results...")
                    for i in range(len(self.viz_data)):
                        var_viz = edict(deepcopy(self.viz_data[i]))
                        var_viz = self.evaluate_batch(opt, var_viz, ep, it, single_gpu=True)
                        self.visualize(opt, var_viz, step=ep, split="eval")
                        self.dump_results(opt, var_viz, ep, train=True)
            
            # collect the eval results into tensors
            for metric_key in metric_eval:
                metric_eval[metric_key] = torch.cat(metric_eval[metric_key], dim=0)

        if opt.world_size > 1:
            metric_gather_dict = {}
            # empty tensors for gathering
            for metric_key in metric_eval:
                metric_gather_dict[metric_key] = [torch.zeros_like(metric_eval[metric_key]).to(opt.device) for _ in range(opt.world_size)]
            
            # gather the metrics
            torch.distributed.barrier()
            for metric_key in metric_eval:
                torch.distributed.all_gather(metric_gather_dict[metric_key], metric_eval[metric_key])
                metric_gather_dict[metric_key] = torch.cat(metric_gather_dict[metric_key], dim=0)
        else:
            metric_gather_dict = metric_eval
            
        # handle last batch, if any
        if len(self.test_loader.sampler) * opt.world_size < len(self.test_data):
            for metric_key in metric_eval:
                metric_gather_dict[metric_key] = [metric_gather_dict[metric_key]]
            for batch in self.aux_test_loader:
                # inference the model
                var = edict(batch)
                var = self.evaluate_batch(opt, var, ep, it, single_gpu=False)

                # record MAE for evaluation
                sample_metrics, var.depth_pred_aligned = self.depth_metric.compute_metrics(
                    var.depth_pred, var.depth_input_map, var.mask_eroded if 'mask_eroded' in var else var.mask_input_map)
                var.rmse = sample_metrics['rmse']
                for metric_key in metric_eval:
                    metric_gather_dict[metric_key].append(sample_metrics[metric_key])
                
                # dump the result if in eval mode
                if not training and opt.device == 0: 
                    self.dump_results(opt, var, ep, write_new=(it == 0))
            
            for metric_key in metric_eval:
                metric_gather_dict[metric_key] = torch.cat(metric_gather_dict[metric_key], dim=0)

        assert metric_gather_dict['l1_err'].shape[0] == len(self.test_data)
        # compute the mean of the metrics
        for metric_key in metric_eval:
            metric_avg[metric_key] = metric_gather_dict[metric_key].mean()
        
        # printout and save the metrics     
        if opt.device == 0:
            # print eval info
            print_eval(opt, depth_metrics=metric_avg)
            val_metric = metric_avg['l1_err']
        
            if training:
                # log/visualize results to tb/vis
                self.log_scalars(opt, var, loss_eval, metric=metric_avg, step=ep, split="eval")
            
            if not training:
                # write to file
                metrics_file = os.path.join(opt.output_path, 'best_val.txt')
                with open(metrics_file, "w") as outfile:
                    for metric_key in metric_avg:
                        outfile.write('{}: {:.6f}\n'.format(metric_key, metric_avg[metric_key].item()))
        
            return val_metric.item()
        return float('inf')

    def evaluate_batch(self, opt, var, ep=None, it=None, single_gpu=False):
        var = util.move_to_device(var, opt.device)
        if single_gpu:
            var  = self.graph.module(opt, var, training=False, get_loss=False)
        else:
            var = self.graph(opt, var, training=False, get_loss=False)
        return var

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        if split=="train":
            sample_metrics, _ = self.depth_metric.compute_metrics(
                var.depth_pred, var.depth_input_map, var.mask_eroded if 'mask_eroded' in var else var.mask_input_map)
            metric = dict(L1_ERR=sample_metrics['l1_err'].mean().item())
        for key, value in loss.items():
            if key=="all": continue
            self.tb.add_scalar("{0}/loss_{1}".format(split, key), value.mean(), step)
        if metric is not None:
            for key, value in metric.items():
                self.tb.add_scalar("{0}/{1}".format(split, key), value, step)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"):
        pass

    @torch.no_grad()
    def dump_results(self, opt, var, ep, write_new=False, train=False):
        # create the dir
        current_folder = "dump" if train == False else "vis_{}".format(ep)
        os.makedirs("{}/{}/".format(opt.output_path, current_folder), exist_ok=True)
        
        # save the results
        util_vis.dump_images(opt, var.idx, "image_input", var.rgb_input_map, masks=None, from_range=(0, 1), folder=current_folder)
        util_vis.dump_images(opt, var.idx, "mask_input", var.mask_input_map, folder=current_folder)
        util_vis.dump_depths(opt, var.idx, "depth_pred", var.depth_pred, var.mask_input_map, rescale=True, folder=current_folder)
        util_vis.dump_depths(opt, var.idx, "depth_input", var.depth_input_map, var.mask_input_map, rescale=True, folder=current_folder)
        if 'seen_points_pred' in var and 'seen_points_gt' in var:
            util_vis.dump_pointclouds_compare(opt, var.idx, "seen_surface", var.seen_points_pred, var.seen_points_gt, folder=current_folder)
        
        if "depth_pred_aligned" in var:
            # get the max and min for the depth map
            batch_size = var.depth_input_map.shape[0]
            mask = var.mask_eroded if 'mask_eroded' in var else var.mask_input_map
            masked_depth_far_bg = var.depth_input_map * mask + (1 - mask) * 1000
            depth_min_gt = masked_depth_far_bg.view(batch_size, -1).min(dim=1)[0]
            masked_depth_invalid_bg = var.depth_input_map * mask + (1 - mask) * 0
            depth_max_gt = masked_depth_invalid_bg.view(batch_size, -1).max(dim=1)[0]
            depth_vis_pred = (var.depth_pred_aligned - depth_min_gt.view(batch_size, 1, 1, 1)) / (depth_max_gt - depth_min_gt).view(batch_size, 1, 1, 1)
            depth_vis_pred = depth_vis_pred * mask + (1 - mask)
            depth_vis_gt = (var.depth_input_map - depth_min_gt.view(batch_size, 1, 1, 1)) / (depth_max_gt - depth_min_gt).view(batch_size, 1, 1, 1)
            depth_vis_gt = depth_vis_gt * mask + (1 - mask)
            util_vis.dump_depths(opt, var.idx, "depth_gt_aligned", depth_vis_gt.clamp(max=1, min=0), None, rescale=False, folder=current_folder)
            util_vis.dump_depths(opt, var.idx, "depth_pred_aligned", depth_vis_pred.clamp(max=1, min=0), None, rescale=False, folder=current_folder)
            if "mask_eroded" in var and "rmse" in var:
                util_vis.dump_images(opt, var.idx, "image_eroded", var.rgb_input_map, masks=var.mask_eroded, metrics=var.rmse, from_range=(0, 1), folder=current_folder)

    def save_checkpoint(self, opt, ep=0, it=0, best_val=np.inf, best_ep=1, latest=False, best=False):
        util.save_checkpoint(opt, self, ep=ep, it=it, best_val=best_val, best_ep=best_ep, latest=latest, best=best)
        if not latest:
            print("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group, opt.name, ep, it))
        if best:
            print("Saving the current model as the best...")
