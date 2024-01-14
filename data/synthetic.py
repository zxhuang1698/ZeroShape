import os
import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
import PIL
import utils.camera as camera

from . import base

class Dataset(base.Dataset):

    def __init__(self, opt, split="train", load_3D=True):
        if split == "test":
            split = "val"
        
        super().__init__(opt, split)
        self.path = "data/train_data"
        self.load_3D = load_3D
        self.subsets = opt.data.synthetic.subset.split(',')
        self.category_dict = {}
        self.category_list = []
        for subset in self.subsets:
            subset_path = "{}/{}".format(self.path, subset)
            categories = [name[:-11] for name in os.listdir("{}/lists".format(subset_path)) if name.endswith("_train.list")]
            self.category_dict[subset] = categories
            self.category_list += [cat for cat in categories]
        
        # for val set, we use only max 10 images per category
        if split == "val": 
            self.max_imgs = 10
            self.data_percentage = 1
        else:
            self.max_imgs = np.inf
            self.data_percentage = opt.data.synthetic.percentage
        
        # dict that map category name to label
        self.cat2label = {}
        # list that map from label to category name
        self.label2cat = []
        accum_idx = 0
        for cat in self.category_list:
            self.cat2label[cat] = accum_idx
            self.label2cat.append(cat)
            accum_idx += 1
        
        # get data list
        self.list = self.get_list(opt, split)
    
    # read the list file, return a list of (category, object_name, sample_id)
    def get_list(self, opt, split):
        data_list = []
        for subset in self.subsets:
            for cat in self.category_dict[subset]:
                list_fname = f"{self.path}/{subset}/lists/{cat}_{split}.list"
                if not os.path.exists(list_fname):
                    continue
                lines = open(list_fname).read().splitlines()
                lines = lines[:round(self.data_percentage*len(lines))]
                for i, img_fname in enumerate(lines):
                    if i >= self.max_imgs: break
                    name = '.'.join(img_fname.split('.')[:-1])
                    object_name = name.split('_')[-2]
                    sample_id = name.split('_')[-1]
                    data_list.append((subset, cat, object_name, sample_id))
        return data_list

    def id_filename_mapping(self, opt, outpath):
        outfile = open(outpath, 'w')
        for i in range(len(self.list)):
            subset, category, object_name, sample_id = self.list[i]
            fname = f"{category}/{category}_{object_name}_{sample_id}"
            image_fname = f"{self.path}/{subset}/images_processed/{fname}.png"
            mask_fname = f"{self.path}/{subset}/masks/{fname}.png"
            pc_name = f"{category}/{category}_{object_name}"
            pc_fname = f"{self.path}/{subset}/pointclouds/{pc_name}.npy"
            outfile.write("{} {} {} {}\n".format(i, image_fname, mask_fname, pc_fname))
        outfile.close()

    def get_image(self, subset, category, object_name, sample_id):
        fname = f"{category}/{category}_{object_name}_{sample_id}"
        image_fname = f"{self.path}/{subset}/images_processed/{fname}.png"
        mask_fname = f"{self.path}/{subset}/masks/{fname}.png"
        image = PIL.Image.open(image_fname).convert("RGB")
        mask = PIL.Image.open(mask_fname).convert("L")
        mask_np = np.array(mask)
        
        #binarize
        mask_np[mask_np <= 50] = 0
        mask_np[mask_np >= 50] = 1.0

        image = PIL.Image.merge("RGBA", (*image.split(), mask))
        bbox = self.get_bbox_from_mask(mask_np, 0.5)
        return image, bbox
    
    def get_depth(self, subset, category, object_name, sample_id):
        fname = f"{category}/{category}_{object_name}_{sample_id}"
        depth_fname = f"{self.path}/{subset}/depth/{fname}.npy"
        depth = torch.tensor(np.load(depth_fname)).unsqueeze(0)
        assert depth.shape[1] == self.opt.H
        mask = 1 - (depth == 0).float()
        return depth, mask

    def get_camera(self, subset, category, object_name, sample_id):
        fname = f"{category}/{category}_{object_name}_{sample_id}"
        intr_p = f"{self.path}/{subset}/camera_data/intr/{fname}.npy"
        extr_p = f"{self.path}/{subset}/camera_data/extr/{fname}.npy"
        Rt = np.load(extr_p)
        K = torch.from_numpy(np.load(intr_p))
        return K, Rt

    def get_pointcloud(self, subset, category, object_name):
        fname = f"{category}/{category}_{object_name}"
        pc_fname = f"{self.path}/{subset}/pointclouds/{fname}.npy"
        pc = np.load(pc_fname)
        dpc = {"points":torch.from_numpy(pc).float()}
        return dpc
    
    def get_gt_sdf(self, subset, category, object_name):
        fname = f"{category}/{category}_{object_name}"
        gt_fname = f"{self.path}/{subset}/gt_sdf/{fname}.npy"
        gt_dict = np.load(gt_fname, allow_pickle=True).item()
        gt_sample_points = torch.from_numpy(gt_dict['sample_pt']).float()
        gt_sample_sdf = torch.from_numpy(gt_dict['sample_sdf']).float() - 0.003
        return gt_sample_points, gt_sample_sdf
    
    def __getitem__(self, idx):
        opt = self.opt
        subset, category, object_name, sample_id = self.list[idx]
        sample = dict(idx=idx)
        sample.update(
            category_label=self.cat2label[category]
        )
        
        # load camera
        K, Rt = self.get_camera(subset, category, object_name, sample_id)
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
        image, bbox = self.get_image(subset, category, object_name, sample_id)
        depth, mask_input_map = self.get_depth(subset, category, object_name, sample_id)
        rgb_input_map = self.preprocess_image(opt, image, bbox)
        sample.update(
            rgb_input_map=rgb_input_map,
            mask_input_map=mask_input_map,
            depth_input_map=depth,
        )
        
        if not self.load_3D:
            return sample

        # load point cloud
        dpc = self.get_pointcloud(subset, category, object_name)
        sample.update(dpc=dpc)
        
        # load gt sdf
        gt_sample_points, gt_sample_sdf = self.get_gt_sdf(subset, category, object_name)
        # sample the sdf points if needed
        if opt.training.n_sdf_points:
            sample_idx = torch.randperm(gt_sample_points.shape[0])[:opt.training.n_sdf_points]
            gt_sample_points = gt_sample_points[sample_idx]
            gt_sample_sdf = gt_sample_sdf[sample_idx]
            
        sample.update(
            gt_sample_points=gt_sample_points,
            gt_sample_sdf=gt_sample_sdf,
        )

        return sample

    def get_1d_bounds(self, arr):
        nz = np.flatnonzero(arr)
        return nz[0], nz[-1]

    def get_bbox_from_mask(self, mask, thr):
        # bbox in xywh
        masks_for_box = (mask > thr).astype(np.float32)
        if masks_for_box.sum() <= 10:
            return None
        x0, x1 = self.get_1d_bounds(masks_for_box.sum(axis=-2))
        y0, y1 = self.get_1d_bounds(masks_for_box.sum(axis=-1))

        return x0, y0, x1, y1

    def preprocess_image(self, opt, image, bbox):
        if image.size[0] != opt.W or image.size[0] != opt.H:
            image = image.resize((opt.W, opt.H))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        return rgb

    def square_crop(self, image, bbox=None, crop_ratio=1.):
        x1, y1, x2, y2 = bbox
        h, w = y2-y1, x2-x1
        yc, xc = (y1+y2)/2, (x1+x2)/2
        S = max(h, w)*1.2
        # crop with random size (cropping out of boundary = padding)
        S2 = S*crop_ratio
        image = torchvision_F.crop(image, top=int(yc-S2/2), left=int(xc-S2/2), height=int(S2), width=int(S2))
        return image

    def __len__(self):
        return len(self.list)
