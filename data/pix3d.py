import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
import PIL
import json
import warnings
import utils.camera as camera
import os

from . import base
from utils.util import EasyDict as edict

class Dataset(base.Dataset):

    def __init__(self, opt, split="train"):
        super().__init__(opt, split)
        
        self.cat_id_all = dict(
            bed='bed', 
            bookcase='bookcase', 
            chair='chair', 
            desk='desk', 
            misc='misc', 
            sofa='sofa', 
            table='table', 
            tool='tool', 
            wardrobe='wardrobe'
        )
        self.max_imgs = opt.data.max_img_cat if opt.data.max_img_cat is not None else np.inf
        # if split != "train": self.max_imgs = 50
        self.cat2label = {}
        accum_idx = 0
        self.cat_id = list(self.cat_id_all.values()) if opt.data.pix3d.cat is None else \
                      [v for k, v in self.cat_id_all.items() if k in opt.data.pix3d.cat.split(",")]
        for cat in self.cat_id:
            self.cat2label[cat] = accum_idx
            accum_idx += 1
        self.label2cat = []
        for cat in self.cat_id:
            key = next(key for key, value in self.cat_id_all.items() if value == cat)
            self.label2cat.append(key) 
            
        self.path = "data/Pix3D"
        self.list = self.get_list(opt, split)
        # self.get_correspondence(opt, split, self.list)
    
    # read the list file, return a list of tuple, (category, sample_name)
    def get_list(self, opt, split):
        cads = []
        for c in self.cat_id:
            list_fname = "data/Pix3D/lists/{}_{}.txt".format(c, split)
            for i, m in enumerate(open(list_fname).read().splitlines()):
                if i >= self.max_imgs: break
                cads.append((c, m))
        return cads
    
    def get_correspondence(self, opt, split, cads):
        cads_eval = []
        cat_names_eval = [name[:-9] for name in os.listdir("data/Pix3D/lists") if name.endswith("_test.txt")]
        for c in cat_names_eval:
            list_fname = "data/Pix3D/lists/{}_{}.txt".format(c, split)
            for i, m in enumerate(open(list_fname).read().splitlines()):
                if i >= self.max_imgs: break
                cads_eval.append((c, m))
    
        # write to file
        if split == "test":
            with open("debug/correspondence.txt", "w") as f:
                for i in range(len(cads)):
                    c, name = cads[i]
                    eval_idx = cads_eval.index((c, name))
                    assert cads_eval[eval_idx] == (c, name)
                    f.write("{}\n".format(eval_idx))

    def id_filename_mapping(self, opt, outpath):
        outfile = open(outpath, 'w')
        for i in range(len(self.list)):
            meta = self.get_metadata(opt, i)
            image_fname = "{0}/{1}".format(self.path, meta.img_path)
            mask_fname = "{0}/{1}".format(self.path, meta.mask_path)
            pc_fname = "{0}/{1}".format(self.path, "pointclouds/" + meta.cad_path[6:])
            pc_fname = pc_fname.replace(".obj", ".npy")
            outfile.write("{} {} {} {}\n".format(i, image_fname, mask_fname, pc_fname))
        outfile.close()
    
    def __getitem__(self, idx):
        opt = self.opt
        sample = dict(idx=idx)
        # load meta
        meta = self.get_metadata(opt, idx)

        # load images and compute distance transform
        image = self.get_image(opt, meta=meta)
        cat_label, _ = self.get_category(opt, idx)
        rgb_input_map, mask_input_map = self.preprocess_image(opt, image)
        sample.update(
            rgb_input_map=rgb_input_map,
            mask_input_map=mask_input_map,
            category_label=cat_label,
        )
        
        # load pose
        intr, pose = self.get_camera(opt, meta=meta)
        sample.update(
            pose_gt=pose,
            intr=intr,
        )
        
        # load GT point cloud (only for validation!)
        dpc = self.get_pointcloud(opt, meta=meta)
        sample.update(dpc=dpc)

        return sample

    def get_image(self, opt, meta):
        image_fname = "{0}/{1}".format(self.path, meta.img_path)
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore")
            image = PIL.Image.open(image_fname).convert("RGB")
        mask_fname = "{0}/{1}".format(self.path, meta.mask_path)
        mask = PIL.Image.open(mask_fname).convert("L")
        image = PIL.Image.merge("RGBA", (*image.split(), mask))
        return image

    def get_category(self, opt, idx):
        c, _ = self.list[idx]
        label = int(self.cat2label[c])
        return label, c

    def preprocess_image(self, opt, image):
        image = image.resize((opt.W, opt.H))
        image = torchvision_F.to_tensor(image)
        rgb, mask = image[:3], image[3:]
        mask = (mask>0.5).float()
        if opt.data.bgcolor is not None:
            # replace background color using mask
            rgb = rgb*mask+opt.data.bgcolor*(1-mask)
        return rgb, mask

    def get_camera(self, opt, meta=None):
        intr = torch.tensor([[1.3875*opt.W, 0, opt.W/2],
                             [0, 1.3875*opt.H, opt.H/2],
                             [0, 0, 1]])
        R = meta.cam.R
        pose_R = camera.pose(R=R)
        pose_T= camera.pose(t=[0, 0, 1.78])
        pose = camera.pose.compose([pose_R, pose_T])
        return intr, pose

    def get_pointcloud(self, opt, meta=None):
        pc_fname = "{0}/{1}".format(self.path, "pointclouds/" + meta.cad_path[6:])
        pc_fname = pc_fname.replace(".obj", ".npy")
        pc = torch.from_numpy(np.load(pc_fname)).float()
        dpc = dict(
            points=pc,
            normals=torch.zeros_like(pc),
        )
        return dpc

    def get_metadata(self, opt, idx, name=None, c=None):
        if name is None or c is None:
            c, name = self.list[idx]
        meta_fname = "{}/annotation/{}/{}.json".format(self.path, c, name)
        meta = json.load(open(meta_fname, "r", encoding='utf-8'))
        img_path = meta["img"].replace("img", "img_processed")
        mask_path = meta["mask"].replace("mask", "mask_processed")
        meta_out = edict(
            cam=edict(
                focal=float(meta["focal_length"]),
                cam_loc=torch.tensor(meta["cam_position"]),
                R=torch.tensor(meta["rot_mat"]),
                T=torch.tensor(meta["trans_mat"]),
            ),
            img_path=img_path,
            mask_path=mask_path,
            cad_path=meta["model"],
            bbox=torch.tensor(meta["bbox"]),
        )
        return meta_out

    def __len__(self):
        return len(self.list)