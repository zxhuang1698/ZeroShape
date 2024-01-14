import os, sys, time
import shutil
import datetime
import torch
import torch.nn.functional as torch_F
import socket
import contextlib
import socket
import torch.distributed as dist
from collections import defaultdict, deque

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def print_eval(opt, loss=None, chamfer=None, depth_metrics=None):
    message = "[eval] "
    if loss is not None: message += "loss:{}".format("{:.3e}".format(loss.all))
    if chamfer is not None:
        message += " chamfer:{}|{}|{}".format("{:.4f}".format(chamfer[0]),
                                                "{:.4f}".format(chamfer[1]),
                                                "{:.4f}".format((chamfer[0]+chamfer[1])/2))
    if depth_metrics is not None:
        for k, v in depth_metrics.items():
            message += "{}:{}, ".format(k, "{:.4f}".format(v))
        message = message[:-2]
    print(message)

def update_timer(opt, timer, ep, it_per_ep):
    momentum = 0.99
    timer.elapsed = time.time()-timer.start
    timer.it = timer.it_end-timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean*momentum+timer.it*(1-momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean*it_per_ep*(opt.max_epoch-ep)

# move tensors to device in-place
def move_to_device(X, device):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = move_to_device(v, device)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = move_to_device(e, device)
    elif isinstance(X, tuple) and hasattr(X, "_fields"): # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd, device)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device, non_blocking=True)
    return X

# detach tensors
def detach_tensors(X):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = detach_tensors(v)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = detach_tensors(e)
    elif isinstance(X, tuple) and hasattr(X, "_fields"): # collections.namedtuple
        dd = X._asdict()
        dd = detach_tensors(dd)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.detach()
    return X

# this recursion seems to only work for the outer loop when dict_type is not dict
def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
    return D

def get_child_state_dict(state_dict, key):
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            param_name = k[7:]
        else:
            param_name = k
        if param_name.startswith("{}.".format(key)):
            out_dict[".".join(param_name.split(".")[1:])] = v
    return out_dict

def resume_checkpoint(opt, model, best):
    load_name = "{0}/best.ckpt".format(opt.output_path) if best else "{0}/latest.ckpt".format(opt.output_path)
    checkpoint = torch.load(load_name, map_location=torch.device(opt.device))
    model.graph.module.load_state_dict(checkpoint["graph"], strict=True)   
    # load the training stats
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched", "scaler"] and key in checkpoint:
            if opt.device == 0: print("restoring {}...".format(key))
            getattr(model, key).load_state_dict(checkpoint[key])
    # also need to record ep, it, best_val if we are returning
    ep, it = checkpoint["epoch"], checkpoint["iter"]
    best_val, best_ep = checkpoint["best_val"], checkpoint["best_ep"] if "best_ep" in checkpoint else 0
    print("resuming from epoch {0} (iteration {1})".format(ep, it))

    return ep, it, best_val, best_ep

def load_checkpoint(opt, model, load_name):
    # load_name as to be given
    checkpoint = torch.load(load_name, map_location=torch.device(opt.device))
    # load individual (possibly partial) children modules
    for name, child in model.graph.module.named_children():
        child_state_dict = get_child_state_dict(checkpoint["graph"], name)
        if child_state_dict:
            if opt.device == 0: print("restoring {}...".format(name))
            child.load_state_dict(child_state_dict, strict=True)
        else:
            if opt.device == 0: print("skipping {}...".format(name))
    return None, None, None, None

def restore_checkpoint(opt, model, load_name=None, resume=False, best=False, evaluate=False):
    # we cannot load and resume at the same time
    assert not (load_name is not None and resume)
    # when resuming we want everything to be the same
    if resume:
        ep, it, best_val, best_ep = resume_checkpoint(opt, model, best)
    # loading is more flexible, as we can only load parts of the model
    else:
        ep, it, best_val, best_ep = load_checkpoint(opt, model, load_name)
    return ep, it, best_val, best_ep

def save_checkpoint(opt, model, ep, it, best_val, best_ep, latest=False, best=False, children=None):
    os.makedirs("{0}/checkpoint".format(opt.output_path), exist_ok=True)
    if isinstance(model.graph, torch.nn.DataParallel) or isinstance(model.graph, torch.nn.parallel.DistributedDataParallel):
        graph = model.graph.module
    else:
        graph = model.graph
    if children is not None:
        graph_state_dict = { k: v for k, v in graph.state_dict().items() if k.startswith(children) }
    else: graph_state_dict = graph.state_dict()
    checkpoint = dict(
        epoch=ep,
        iter=it,
        best_val=best_val,
        best_ep=best_ep,
        graph=graph_state_dict,
    )
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched", "scaler"]:
            checkpoint.update({key: getattr(model, key).state_dict()})
    torch.save(checkpoint, "{0}/latest.ckpt".format(opt.output_path))
    if best:
        shutil.copy("{0}/latest.ckpt".format(opt.output_path),
                    "{0}/best.ckpt".format(opt.output_path))
    if not latest:
        shutil.copy("{0}/latest.ckpt".format(opt.output_path),
                    "{0}/checkpoint/ep{1}.ckpt".format(opt.output_path, ep))

def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open

def get_layer_dims(layers):
    # return a list of tuples (k_in, k_out)
    return list(zip(layers[:-1], layers[1:]))

@contextlib.contextmanager
def suppress(stdout=False, stderr=False):
    with open(os.devnull, "w") as devnull:
        if stdout: old_stdout, sys.stdout = sys.stdout, devnull
        if stderr: old_stderr, sys.stderr = sys.stderr, devnull
        try: yield
        finally:
            if stdout: sys.stdout = old_stdout
            if stderr: sys.stderr = old_stderr

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_outs, x_in):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg += grad_dout2.view(batch_size, -1).sum(1)
    return reg / len(d_outs)

# import matplotlib.pyplot as plt
def interpolate_depth(depth_input, mask_input, size, bg_depth=20):
    assert len(depth_input.shape) == len(mask_input.shape) == 4
    mask = (mask_input > 0.5).float()
    depth_valid = depth_input * mask
    depth_valid = torch_F.interpolate(depth_valid, size, mode='bilinear', align_corners=False)
    mask = torch_F.interpolate(mask, size, mode='bilinear', align_corners=False)
    depth_out = depth_valid / (mask + 1.e-6)
    mask_binary = (mask > 0.5).float()
    depth_out = depth_out * mask_binary + bg_depth * (1 - mask_binary)
    return depth_out, mask_binary

# import matplotlib.pyplot as plt
# import torchvision
def interpolate_coordmap(coord_map, mask_input, size, bg_coord=0):
    assert len(coord_map.shape) == len(mask_input.shape) == 4
    mask = (mask_input > 0.5).float()
    coord_valid = coord_map * mask
    coord_valid = torch_F.interpolate(coord_valid, size, mode='bilinear', align_corners=False)
    mask = torch_F.interpolate(mask, size, mode='bilinear', align_corners=False)
    coord_out = coord_valid / (mask + 1.e-6)
    mask_binary = (mask > 0.5).float()
    coord_out = coord_out * mask_binary + bg_coord * (1 - mask_binary)
    return coord_out, mask_binary

def cleanup():
    dist.destroy_process_group()

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def setup(rank, world_size, port_no):
    full_address = 'tcp://127.0.0.1:' + str(port_no)
    dist.init_process_group("nccl", init_method=full_address, rank=rank, world_size=world_size)

def print_grad(grad, prefix=''):
    print("{} --- Grad Abs Mean, Grad Max, Grad Min: {:.5f} | {:.5f} | {:.5f}".format(prefix, grad.abs().mean().item(), grad.max().item(), grad.min().item()))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
        
class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)

