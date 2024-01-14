import numpy as np
import os, sys, time
import torch
import random
import string
import yaml
import utils.util as util
import time

from utils.util import EasyDict as edict

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

def parse_arguments(args):
    # parse from command line (syntax: --key1.key2.key3=value)
    opt_cmd = {}
    for arg in args:
        assert(arg.startswith("--"))
        if "=" not in arg[2:]: # --key means key=True, --key! means key=False
            key_str, value = (arg[2:-1], "false") if arg[-1]=="!" else (arg[2:], "true")
        else:
            key_str, value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub: opt_sub[k] = {}
            opt_sub = opt_sub[k]
        # if opt_cmd['key1']['key2']['key3'] already exist for key1.key2.key3, print key3 as error msg 
        assert keys_sub[-1] not in opt_sub, keys_sub[-1]
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    opt_cmd = edict(opt_cmd)
    return opt_cmd

def set(opt_cmd={}, verbose=True, safe_check=True):
    print("setting configurations...")
    fname = opt_cmd.yaml # load from yaml file
    opt_base = load_options(fname)
    # override with command line arguments
    opt = override_options(opt_base, opt_cmd, key_stack=[], safe_check=safe_check)
    process_options(opt)
    if verbose:
        def print_options(opt, level=0):
            for key, value in sorted(opt.items()):
                if isinstance(value, (dict, edict)):
                    print("   "*level+"* "+key+":")
                    print_options(value, level+1)
                else:
                    print("   "*level+"* "+key+":", value)
        print_options(opt)
    return opt

def load_options(fname):
    with open(fname) as file:
        opt = edict(yaml.safe_load(file))
    if "_parent_" in opt:
        # load parent yaml file(s) as base options
        parent_fnames = opt.pop("_parent_")
        if type(parent_fnames) is str:
            parent_fnames = [parent_fnames]
        for parent_fname in parent_fnames:
            opt_parent = load_options(parent_fname)
            opt_parent = override_options(opt_parent, opt, key_stack=[])
            opt = opt_parent
    print("loading {}...".format(fname))
    return opt

def override_options(opt, opt_over, key_stack=None, safe_check=False):
    for key, value in opt_over.items():
        if isinstance(value, dict):
            # parse child options (until leaf nodes are reached)
            opt[key] = override_options(opt.get(key, dict()), value, key_stack=key_stack+[key], safe_check=safe_check)
        else:
            # ensure command line argument to override is also in yaml file
            if safe_check and key not in opt:
                add_new = None
                while add_new not in ["y", "n"]:
                    key_str = ".".join(key_stack+[key])
                    add_new = input("\"{}\" not found in original opt, add? (y/n) ".format(key_str))
                if add_new=="n":
                    print("safe exiting...")
                    exit()
            opt[key] = value
    return opt

def process_options(opt):
    # set seed
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    else:
        # create random string as run ID
        randkey = "".join(random.choice(string.ascii_uppercase) for _ in range(4))
        opt.name += "_{}".format(randkey)
    # other default options
    opt.output_path = "{0}/{1}/{2}".format(opt.output_root, opt.group, opt.name)
    os.makedirs(opt.output_path, exist_ok=True)
    assert torch.cuda.is_available()
    opt.device = "cuda:{}".format(opt.gpu)
    opt.H, opt.W = opt.image_size
    if opt.freq.eval is None:
        opt.freq.eval = max(opt.max_epoch // 20, 1) 
    if 'loss_weight' in opt:
        opt.get_depth = False
        opt.get_normal = False

def save_options_file(opt):
    opt_fname = "{}/options.yaml".format(opt.output_path)
    if os.path.isfile(opt_fname):
        with open(opt_fname) as file:
            opt_old = yaml.safe_load(file)
        if opt!=opt_old:
            # prompt if options are not identical
            opt_new_fname = "{}/options_temp.yaml".format(opt.output_path)
            with open(opt_new_fname, "w") as file:
                yaml.safe_dump(util.to_dict(opt), file, default_flow_style=False, indent=4)
            print("existing options file found (different from current one)...")
            os.system("diff {} {}".format(opt_fname, opt_new_fname))
            os.system("rm {}".format(opt_new_fname))
            if not opt.debug:
                print("please cancel within 10 seconds if you do not want to override...")
                time.sleep(10)
        else: print("existing options file found (identical)")
    else: print("(creating new options file...)")
    with open(opt_fname, "w") as file:
        yaml.safe_dump(util.to_dict(opt), file, default_flow_style=False, indent=4)
