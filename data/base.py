import torch
from copy import deepcopy

class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, split):
        super().__init__()
        self.opt = deepcopy(opt)
        self.split = split

    def setup_loader(self, opt, shuffle=False, drop_last=True, subcat=None, batch_size=None, use_ddp=True):
        sampler = torch.utils.data.distributed.DistributedSampler(self,
            num_replicas=opt.world_size, rank=opt.device, shuffle=shuffle, drop_last=drop_last
        ) if (use_ddp and 'world_size' in opt) else None
        if batch_size is None: batch_size=opt.batch_size
        loader = torch.utils.data.DataLoader(self,
            batch_size=batch_size,
            num_workers=opt.data.num_workers,
            shuffle=shuffle if sampler is None else False,
            drop_last=drop_last if self.split == 'train' else False,
            pin_memory=True,
            sampler=sampler
        )
        if opt.device == 0:
            print("number of samples: {}".format(len(self)))
        return loader

    def get_list(self, opt):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_image(self, opt, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.list)
