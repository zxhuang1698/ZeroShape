import os, sys, random
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
    evaluator = engine.Runner(opt)
    evaluator.load_dataset(opt)
    evaluator.test_data.id_filename_mapping(opt, os.path.join(opt.output_path, 'data_list.txt'))
    evaluator.build_networks(opt)
    evaluator.restore_checkpoint(opt, best=True, evaluate=True)
    evaluator.setup_visualizer(opt, test=True)

    evaluator.evaluate(opt, ep=0)

def main():
    print("[{}] (evaluating)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    opt.eval.n_vis = 1

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