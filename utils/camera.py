# partially from https://github.com/chenhsuanlin/signed-distance-SRN

import numpy as np
import torch

class Pose():
    # a pose class with util methods
    def __call__(self, R=None, t=None):
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t, torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor): R = torch.tensor(R)
            if not isinstance(t, torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3, 3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1) # [..., 3, 4]
        assert(pose.shape[-2:]==(3, 4))
        return pose

    def invert(self, pose, use_inverse=False):
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # pose_new(x) = poseN(...(pose2(pose1(x)))...)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b(pose_a(x))
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

pose = Pose()

# unit sphere normalization
def valid_norm_fac(seen_points, mask):
    '''
    seen_points: [B, H*W, 3]
    mask: [B, 1, H, W], boolean
    '''
    # get valid points
    batch_size = seen_points.shape[0]
    # [B, H*W]
    mask = mask.view(batch_size, seen_points.shape[1])
    
    # get mean and variance by sample
    means, max_dists = [], []
    for b in range(batch_size):
        # [N_valid, 3]
        seen_points_valid = seen_points[b][mask[b]]
        # [3]
        xyz_mean = torch.mean(seen_points_valid, dim=0)
        seen_points_valid_zmean = seen_points_valid - xyz_mean
        # scalar
        max_dist = torch.max(seen_points_valid_zmean.norm(dim=1))
        means.append(xyz_mean)
        max_dists.append(max_dist)
    # [B, 3]
    means = torch.stack(means, dim=0)
    # [B]
    max_dists = torch.stack(max_dists, dim=0)
    return means, max_dists

def get_pixel_grid(opt, H, W):
    y_range = torch.arange(H, dtype=torch.float32).to(opt.device)
    x_range = torch.arange(W, dtype=torch.float32).to(opt.device)
    Y, X = torch.meshgrid(y_range, x_range, indexing='ij')
    Z = torch.ones_like(Y)
    xyz_grid = torch.stack([X, Y, Z],dim=-1).view(-1,3) 
    return xyz_grid

def unproj_depth(opt, depth, intr):
    '''
    depth: [B, 1, H, W]
    intr: [B, 3, 3]
    '''
    batch_size, _, H, W = depth.shape
    assert opt.H == H == W
    depth = depth.squeeze(1)
    
    # [B, 3, 3]
    K_inv = torch.linalg.inv(intr).float()
    # [1, H*W,3]
    pixel_grid = get_pixel_grid(opt, H, W).unsqueeze(0)
    # [B, H*W,3]
    pixel_grid = pixel_grid.repeat(batch_size, 1, 1)
    # [B, 3, H*W]
    ray_dirs = K_inv @ pixel_grid.permute(0, 2, 1).contiguous()
    # [B, H*W, 3], in camera coordinates
    seen_points = ray_dirs.permute(0, 2, 1).contiguous() * depth.view(batch_size, H*W, 1)

    return seen_points

def to_hom(X):
    '''
    X: [B, N, 3]
    Returns:
        X_hom: [B, N, 4]
    '''
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom

def world2cam(X_world, pose):
    '''
    X_world: [B, N, 3]
    pose: [B, 3, 4]
    Returns:
        X_cam: [B, N, 3]
    '''
    X_hom = to_hom(X_world)
    X_cam = X_hom @ pose.transpose(-1, -2)
    return X_cam

def cam2img(X_cam, cam_intr):
    '''
    X_cam: [B, N, 3]
    cam_intr: [B, 3, 3]
    Returns:
        X_img: [B, N, 3]
    '''
    X_img = X_cam @ cam_intr.transpose(-1, -2)
    return X_img

def proj_points(opt, points, intr, pose):
    '''
    points: [B, N, 3]
    intr: [B, 3, 3]
    pose: [B, 3, 4]
    '''
    # [B, N, 3]
    points_cam = world2cam(points, pose)
    # [B, N]
    depth = points_cam[..., 2]
    # [B, N, 3]
    points_img = cam2img(points_cam, intr)
    # [B, N, 2]
    points_2D = points_img[..., :2] / points_img[..., 2:]
    return points_2D, depth

def azim_to_rotation_matrix(azim, representation='angle'):
    """Azim is angle with vector +X, rotated in XZ plane"""
    if representation == 'rad':
        # [B, ]
        cos, sin = torch.cos(azim), torch.sin(azim)
    elif representation == 'angle':
        # [B, ]
        azim = azim * np.pi / 180
        cos, sin = torch.cos(azim), torch.sin(azim)
    elif representation == 'trig':
        # [B, 2]
        cos, sin = azim[:, 0], azim[:, 1]
    R = torch.eye(3, device=azim.device)[None].repeat(len(azim), 1, 1)
    zeros = torch.zeros(len(azim), device=azim.device)
    R[:, 0, :] = torch.stack([cos, zeros, sin], dim=-1)
    R[:, 2, :] = torch.stack([-sin, zeros, cos], dim=-1)
    return R

def elev_to_rotation_matrix(elev, representation='angle'):
    """Angle with vector +Z in YZ plane"""
    if representation == 'rad':
        # [B, ]
        cos, sin = torch.cos(elev), torch.sin(elev)
    elif representation == 'angle':
        # [B, ]
        elev = elev * np.pi / 180
        cos, sin = torch.cos(elev), torch.sin(elev)
    elif representation == 'trig':
        # [B, 2]
        cos, sin = elev[:, 0], elev[:, 1]
    R = torch.eye(3, device=elev.device)[None].repeat(len(elev), 1, 1)
    R[:, 1, 1:] = torch.stack([cos, -sin], dim=-1)
    R[:, 2, 1:] = torch.stack([sin, cos], dim=-1)
    return R

def roll_to_rotation_matrix(roll, representation='angle'):
    """Angle with vector +X in XY plane"""
    if representation == 'rad':
        # [B, ]
        cos, sin = torch.cos(roll), torch.sin(roll)
    elif representation == 'angle':
        # [B, ]
        roll = roll * np.pi / 180
        cos, sin = torch.cos(roll), torch.sin(roll)
    elif representation == 'trig':
        # [B, 2]
        cos, sin = roll[:, 0], roll[:, 1]
    R = torch.eye(3, device=roll.device)[None].repeat(len(roll), 1, 1)
    R[:, 0, :2] = torch.stack([cos, sin], dim=-1)
    R[:, 1, :2] = torch.stack([-sin, cos], dim=-1)
    return R

def get_rotation_sphere(azim_sample=4, elev_sample=4, roll_sample=4, scales=[1.0], device='cuda'):
    rotations = []
    azim_range = [0, 360]
    elev_range = [0, 360]
    roll_range = [0, 360]
    azims = np.linspace(azim_range[0], azim_range[1], num=azim_sample, endpoint=False)
    elevs = np.linspace(elev_range[0], elev_range[1], num=elev_sample, endpoint=False)
    rolls = np.linspace(roll_range[0], roll_range[1], num=roll_sample, endpoint=False)
    for scale in scales:
        for azim in azims:
            for elev in elevs:
                for roll in rolls:
                    Ry = azim_to_rotation_matrix(torch.tensor([azim]))
                    Rx = elev_to_rotation_matrix(torch.tensor([elev]))
                    Rz = roll_to_rotation_matrix(torch.tensor([roll]))
                    R_permute = torch.tensor([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, -1, 0]
                    ]).float().to(Ry.device).unsqueeze(0).expand_as(Ry)
                    R = scale * Rz@Rx@Ry@R_permute
                    rotations.append(R.to(device).float())
    return torch.cat(rotations, dim=0)