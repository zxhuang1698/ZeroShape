import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
import PIL
import utils.camera as camera
import os
import cv2

from . import base

class Dataset(base.Dataset):

    def __init__(self, opt, split="train", load_3D=True):
        super().__init__(opt, split)
        
        self.path = "data/OmniObject3D"
        self.opt = opt
        self.load_3D = load_3D
        self.cat_names = [name[:-10] for name in os.listdir(os.path.join(self.path, "lists")) if name.endswith("_test.list")]
        
        # mapping from category name to label
        accum_idx = 0
        self.cat2label = {}
        for cat in self.cat_names:
            self.cat2label[cat] = accum_idx
            accum_idx += 1
            
        # mapping from label to category name
        self.label2cat = self.cat_names
        
        # get the list of all images
        self.list = self.get_list(opt, "test")
    
    # read the list file, return a list of tuple, (category, sample_name)
    def get_list(self, opt, split):
        cads = []
        for c in self.cat_names:
            list_fname = "{}/lists/{}_{}.list".format(self.path, c, split)
            for i, image_name in enumerate(open(list_fname).read().splitlines()):
                sample_name = image_name.split('.')[0]
                cads.append((c, sample_name))
        return cads

    def id_filename_mapping(self, opt, outpath):
        outfile = open(outpath, 'w')
        for i in range(len(self.list)):
            category, name = self.list[i]
            image_fname = "{}/images_processed/{}/{}.png".format(self.path, category, name)
            mask_fname = "{}/masks_processed/{}/{}.png".format(self.path, category, name)
            pc_fname = "{}/pointclouds/{}/{}.npy".format(self.path, category, name)
            outfile.write("{} {} {} {}\n".format(i, image_fname, mask_fname, pc_fname))
        outfile.close()

    def get_camera(self, idx):
        category, name = self.list[idx]
        extr_fname = "{}/camera_data/extr/{}/{}.npy".format(self.path, category, name)
        Rt = torch.tensor(np.load(extr_fname)).float()
        f = 1.3875
        K = torch.tensor([[f*self.opt.W, 0, self.opt.W/2],
                        [0, f*self.opt.H, self.opt.H/2],
                        [0, 0, 1]]).float()
        return K, Rt

    def get_1d_bounds(self, arr):
        nz = np.flatnonzero(arr)
        return nz[0], nz[-1]

    def get_bbox_from_mask(self, mask, thr=0.5):
        # bbox in xywh
        mask_np = np.array(mask)
        masks_for_box = (mask_np >= thr).astype(np.float32)
        if masks_for_box.sum() <= 10:
            return None
        x0, x1 = self.get_1d_bounds(masks_for_box.sum(axis=-2))
        y0, y1 = self.get_1d_bounds(masks_for_box.sum(axis=-1))
        return x0, y0, x1, y1

    def get_image(self, idx):
        category, name = self.list[idx]
        image_fname = "{}/images_processed/{}/{}.png".format(self.path, category, name)
        image = PIL.Image.open(image_fname).convert("RGB")
        return image

    def get_depth(self, idx):
        category, name = self.list[idx]
        depth_fname = "{}/depth/{}/{}.npy".format(self.path, category, name)
        depth = torch.tensor(np.load(depth_fname)).float().unsqueeze(0)
        assert depth.shape[1] == self.opt.H
        mask = 1 - (depth == 0).float()
        return depth, mask

    def preprocess_image(self, image, mask, bbox):
        opt = self.opt
        if image.size[0] != opt.W or image.size[1] != opt.H:
            image = image.resize((opt.W, opt.H))
        rgb = torchvision_F.to_tensor(image)
        if opt.data.bgcolor is not None:
            # replace background color using mask
            rgb = rgb * mask + opt.data.bgcolor * (1 - mask)
        return rgb

    def erode_mask(self, mask, iterations=5):
        # [H, W]
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        mask_eroded = cv2.erode(mask_np, np.ones((3,3), np.uint8), iterations=iterations)
        mask_eroded = torch.tensor(mask_eroded).unsqueeze(0).float()
        if mask_eroded.sum() == 0:
            mask_eroded = self.erode_mask(mask, iterations=iterations-1)
        return mask_eroded

    def square_crop(self, opt, image, bbox=None, crop_ratio=1.):
        # crop to canonical image size
        x1, y1, x2, y2 = bbox
        h, w = y2-y1, x2-x1
        yc, xc = (y1+y2)/2, (x1+x2)/2
        S = max(h, w)*1.2
        # crop with random size (cropping out of boundary = padding)
        S2 = S*crop_ratio
        image = torchvision_F.crop(image, top=int(yc-S2/2), left=int(xc-S2/2), height=int(S2), width=int(S2))
        return image

    def get_pointcloud(self, idx):
        category, name = self.list[idx]
        pc_fname = "{}/pointclouds/{}/{}.npy".format(self.path, category, '_'.join(name.split('_')[:-1]))
        pc = np.load(pc_fname)
        dpc = {"points":torch.from_numpy(pc).float()}
        return dpc

    def __getitem__(self, idx):
        category, _ = self.list[idx]
        sample = dict(
            idx=idx,
            category_label=self.cat2label[category]
        )
        # load camera
        K, Rt = self.get_camera(idx)
        R = np.zeros((3,4))
        R[:3,:3] = Rt[:3,:3]
        t = Rt[:3,3]
        t = camera.pose(t=t)
        pose = camera.pose.compose([R, t])
        sample.update(
            pose_gt=pose.float(),
            intr=K.float(),
        )
        
        # load images
        image = self.get_image(idx)
        depth, mask = self.get_depth(idx)
        bbox = self.get_bbox_from_mask(mask)
        rgb = self.preprocess_image(image, mask, bbox)
        sample.update(
            rgb_input_map=rgb,
            mask_input_map=mask,
            depth_input_map=depth,
        )
        
        if not self.load_3D:
            return sample
        
        # load point cloud
        dpc = self.get_pointcloud(idx)
        sample.update(dpc=dpc)

        return sample

    def __len__(self):
        return len(self.list)