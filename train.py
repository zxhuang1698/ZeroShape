import numpy as np
import os, sys, time, random
import torch
import torch.multiprocessing as mp

import utils.options as options
from utils.util import is_port_in_use
import importlib

def main_worker(rank, world_size, port, opt):

    opt.device = rank
    opt.world_size = world_size
    opt.port = port
    
    engine = importlib.import_module('model.{}_engine'.format(opt.yaml.split('/')[-1].split('.')[0]))
    trainer = engine.Runner(opt)

    trainer.load_dataset(opt)
    trainer.build_networks(opt)
    trainer.setup_optimizer(opt)
    trainer.restore_checkpoint(opt)
    trainer.setup_visualizer(opt)

    trainer.train(opt)

def main():
    print("[{}] (training)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)
    
    if opt.arch.depth.pretrained and not os.path.exists(opt.arch.depth.pretrained):
        os.makedirs(os.path.dirname(opt.arch.depth.pretrained), exist_ok=True)
        os.system("wget -O {} https://www.dropbox.com/s/bua998sjhdizn6b/omnidata_dpt_depth_v2.ckpt?dl=0".format(opt.arch.depth.pretrained))

    port = (os.getpid() % 32000) + 32768
    while is_port_in_use(port):
        port += 1
    world_size = torch.cuda.device_count()
    if world_size == 1:
        main_worker(0, world_size, port, opt)
    else:
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, port, opt))

if __name__ == "__main__":
    main()