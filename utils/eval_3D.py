import numpy as np
import torch
import threading
import mcubes
import trimesh
from external.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.util_vis import show_att_on_image
from utils.camera import get_rotation_sphere

@torch.no_grad()
def get_dense_3D_grid(opt, var, N=None):
    batch_size = len(var.idx)
    N = N or opt.eval.vox_res
    # -0.6, 0.6
    range_min, range_max = opt.eval.range
    grid = torch.linspace(range_min, range_max, N+1, device=opt.device)
    points_3D = torch.stack(torch.meshgrid(grid, grid, grid, indexing='ij'), dim=-1) # [N, N, N, 3]
    # actually N+1 instead of N
    points_3D = points_3D.repeat(batch_size, 1, 1, 1, 1) # [B, N, N, N, 3]
    return points_3D

@torch.no_grad()
def compute_level_grid(opt, impl_network, latent_depth, latent_semantic, points_3D, images, vis_attn=False):
    # needed for amp
    latent_depth = latent_depth.to(torch.float32) if latent_depth is not None else None
    latent_semantic = latent_semantic.to(torch.float32) if latent_semantic is not None else None
    
    # process points in sliced way
    batch_size = points_3D.shape[0]
    N = points_3D.shape[1]
    assert N == points_3D.shape[2] == points_3D.shape[3]
    assert points_3D.shape[4] == 3
    
    points_3D = points_3D.view(batch_size, N, N*N, 3)
    occ = []
    attn = []
    for i in range(N):
        # [B, N*N, 3]
        points_slice = points_3D[:, i]
        # [B, N*N, 3] -> [B, N*N], [B, N*N, 1+feat_res**2]
        occ_slice, attn_slice = impl_network(latent_depth, latent_semantic, points_slice)
        occ.append(occ_slice)
        attn.append(attn_slice.detach())
    # [B, N, N*N] -> [B, N, N, N]
    occ = torch.stack(occ, dim=1).view(batch_size, N, N, N)
    occ = torch.sigmoid(occ)
    if vis_attn:
        N_global = 1
        feat_res = opt.H // opt.arch.win_size
        attn = torch.stack(attn, dim=1).view(batch_size, N, N, N, N_global+feat_res**2)
        # average along Z, [B, N, N, N_global+feat_res**2]
        attn = torch.mean(attn, dim=3)
        # [B, N, N, N_global] -> [B, N, N, 1]
        attn_global = attn[:, :, :, :N_global].sum(dim=-1, keepdim=True)
        # [B, N, N, feat_res, feat_res]
        attn_local = attn[:, :, :, N_global:].view(batch_size, N, N, feat_res, feat_res)
        # [B, N, N, feat_res, feat_res]
        attn_vis = attn_global.unsqueeze(-1) + attn_local
        # list of frame lists
        images_vis = []
        for b in range(batch_size):
            images_vis_sample = []
            for row in range(0, N, 8):
                if row % 16 == 0:
                    col_range = range(0, N//8*8+1, 8)
                else:
                    col_range = range(N//8*8, -1, -8)
                for col in col_range:
                    # [feat_res, feat_res], x is col
                    attn_curr = attn_vis[b, col, row]
                    attn_curr = torch.nn.functional.interpolate(
                        attn_curr.unsqueeze(0).unsqueeze(0), size=(opt.H, opt.W), 
                        mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0).cpu().numpy()
                    attn_curr /= attn_curr.max()
                    # [feat_res, feat_res, 3]
                    image_curr = images[b].permute(1, 2, 0).cpu().numpy()
                    # merge the image and the attention
                    images_vis_sample.append(show_att_on_image(image_curr, attn_curr))
            images_vis.append(images_vis_sample)
    return occ, images_vis if vis_attn else None

@torch.no_grad()
def standardize_pc(pc):
    assert len(pc.shape) == 3
    pc_mean = pc.mean(dim=1, keepdim=True) 
    pc_zmean = pc - pc_mean
    origin_distance = (pc_zmean**2).sum(dim=2, keepdim=True).sqrt()
    scale = torch.sqrt(torch.sum(origin_distance**2, dim=1, keepdim=True) / pc.shape[1])
    pc_standardized = pc_zmean / (scale * 2)
    return pc_standardized

@torch.no_grad()
def normalize_pc(pc):
    assert len(pc.shape) == 3
    pc_mean = pc.mean(dim=1, keepdim=True) 
    pc_zmean = pc - pc_mean
    length_x = pc_zmean[:, :, 0].max(dim=-1)[0] - pc_zmean[:, :, 0].min(dim=-1)[0]
    length_y = pc_zmean[:, :, 1].max(dim=-1)[0] - pc_zmean[:, :, 1].min(dim=-1)[0]
    length_max = torch.stack([length_x, length_y], dim=-1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    pc_normalized = pc_zmean / (length_max + 1.e-7)
    return pc_normalized

@torch.no_grad()
def eval_metrics_default(opt, var, impl_network, vis_only=False):
    points_3D = get_dense_3D_grid(opt, var) # [B, N, N, N, 3]
    batch_size = points_3D.shape[0]
    level_vox, attn_vis = compute_level_grid(opt, impl_network, var.latent_depth, var.latent_semantic, 
                                             points_3D, var.rgb_input_map, vis_only)
    if attn_vis:
        var.attn_vis = attn_vis
    var.eval_vox = points_3D.view(batch_size, -1, 3)
    # level_grids: a list of length B, each is [N, N, N]
    *level_grids, = level_vox.cpu().numpy()
    meshes,pointclouds = convert_to_explicit(opt,level_grids,isoval=0.5,to_pointcloud=True)
    var.mesh_pred = meshes# colorize_mesh(opt, var.volumes, meshes) if vis_only else meshes
    var.dpc_pred = torch.tensor(pointclouds, dtype=torch.float32, device=opt.device)

    # transform the gt to view-centered frame
    R_gt = var.pose_gt[..., :3]
    var.dpc.points = (R_gt @ var.dpc.points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
    if opt.data.dataset_test == 'pix3d':
        var.dpc.points[:, :, :2] *= -1

    var.dpc_pred = normalize_pc(var.dpc_pred)
    var.dpc.points = normalize_pc(var.dpc.points)

    if vis_only: return
    if opt.eval.icp:
        var.dpc_pred = ICP(opt, var.dpc_pred, var.dpc.points)
    dist_acc, dist_comp, _, _ = chamfer_distance(opt, X1=var.dpc_pred, X2=var.dpc.points)
    var.f_score = compute_fscore(dist_acc, dist_comp, opt.eval.f_thresholds)   # [B, n_threshold]
    # dist_acc: [B, n_points_pred]
    # dist_comp: [B, n_points_gt]
    assert dist_acc.shape[1] == opt.eval.num_points
    var.cd_acc = dist_acc.mean(dim=1)
    var.cd_comp = dist_comp.mean(dim=1)
    return dist_acc.mean(), dist_comp.mean()

def brute_force_search(pc_pred, pc_gt, f_thresholds=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2], device="cuda"):
    pc_pred = pc_pred.to(device).unsqueeze(0).float()
    pc_gt = pc_gt.to(device).unsqueeze(0).float().contiguous()
    pc_gt = normalize_pc(pc_gt)
    
    # brute force evaluation
    # get the best CD by iterating over the pose
    best_cd = np.inf
    rotations = get_rotation_sphere(azim_sample=24, elev_sample=24, roll_sample=12, scales=[1.0])
    batch_size = 24
    
    # process the pointcloud in batches
    for i in range(0, len(rotations), batch_size):
        rotation_batch = rotations[i:i+batch_size].to(device)
        pc_pred_rotated = (rotation_batch @ pc_pred.repeat(rotation_batch.shape[0], 1, 1).permute(0, 2, 1)).permute(0, 2, 1)
        pc_pred_rotated = normalize_pc(pc_pred_rotated).contiguous()
        acc, comp, _, _ = chamfer_distance(None, pc_pred_rotated, pc_gt.repeat(rotation_batch.shape[0], 1, 1).contiguous())
        f_score = compute_fscore(acc, comp, f_thresholds)
        acc, comp = acc.mean(dim=1), comp.mean(dim=1)
        cd = (acc + comp) / 2
        # save the best cd
        for j in range(len(cd)):
            if cd[j] < best_cd:
                best_pc_pred = pc_pred_rotated[j].clone()
                best_acc = acc[j]
                best_comp = comp[j]
                best_cd = cd[j]
                best_fscore = f_score[j]
                best_rotation = rotation_batch[j].clone()

    return best_acc, best_comp, best_fscore, best_pc_pred, pc_gt

def eval_metrics_BF(opt, var, impl_network, vis_only=False):
    points_3D = get_dense_3D_grid(opt, var) # [B, N, N, N, 3]
    batch_size = points_3D.shape[0]
    level_vox, attn_vis = compute_level_grid(opt, impl_network, var.latent_depth, var.latent_semantic, 
                                             points_3D, var.rgb_input_map, vis_only)
    if attn_vis:
        var.attn_vis = attn_vis
    var.eval_vox = points_3D.view(batch_size, -1, 3)
    # level_grids: a list of length B, each is [N, N, N]
    *level_grids, = level_vox.cpu().numpy()
    meshes, pointclouds = convert_to_explicit(opt, level_grids, isoval=0.5, to_pointcloud=True)
    var.mesh_pred = meshes
    var.dpc_pred = torch.tensor(pointclouds, dtype=torch.float32, device=opt.device)

    # transform the gt to view-centered frame
    R_gt = var.pose_gt[..., :3]
    var.dpc.points = (R_gt @ var.dpc.points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
    if opt.data.dataset_test == 'pix3d':
        var.dpc.points[:, :, :2] *= -1
    
    if vis_only: return
    # perform per-sample alignment
    cd_acc, cd_comp, f_score = [], [], []
    for i in range(batch_size):
        best_acc, best_comp, best_fscore, best_pred, best_gt = \
            brute_force_search(var.dpc_pred[i], var.dpc.points[i], opt.eval.f_thresholds, opt.device)
        # rotate the prediction
        var.dpc_pred[i] = best_pred.clone()
        var.dpc.points[i] = best_gt.clone()
        cd_acc.append(best_acc)
        cd_comp.append(best_comp)
        f_score.append(best_fscore)
    var.cd_acc = torch.stack(cd_acc, dim=0)
    var.cd_comp = torch.stack(cd_comp, dim=0)
    var.f_score = torch.stack(f_score, dim=0)
    return var.cd_acc.mean(), var.cd_comp.mean()

def eval_metrics(opt, var, impl_network, vis_only=False):
    if opt.eval.brute_force:
        return eval_metrics_BF(opt, var, impl_network, vis_only)
    else:
        return eval_metrics_default(opt, var, impl_network, vis_only)

def compute_fscore(dist1, dist2, thresholds=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2]):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscores
    """
    fscores = []
    for threshold in thresholds:
        precision = torch.mean((dist1 < threshold).float(), dim=1)  # [B, ]
        recall = torch.mean((dist2 < threshold).float(), dim=1)
        fscore = 2 * precision * recall / (precision + recall)
        fscore[torch.isnan(fscore)] = 0
        fscores.append(fscore)
    fscores = torch.stack(fscores, dim=1)
    return fscores

def convert_to_explicit(opt, level_grids, isoval=0., to_pointcloud=False):
    N = len(level_grids)
    meshes = [None]*N
    pointclouds = [None]*N if to_pointcloud else None
    threads = [threading.Thread(target=convert_to_explicit_worker,
                                args=(opt, i, level_grids[i], isoval, meshes),
                                kwargs=dict(pointclouds=pointclouds),
                                daemon=False) for i in range(N)]
    for t in threads: t.start()
    for t in threads: t.join()
    if to_pointcloud:
        pointclouds = np.stack(pointclouds, axis=0)
        return meshes, pointclouds
    else: return meshes

def convert_to_explicit_worker(opt, i, level_vox_i, isoval, meshes, pointclouds=None):
    # use marching cubes to convert implicit surface to mesh
    vertices, faces = mcubes.marching_cubes(level_vox_i, isovalue=isoval)
    assert(level_vox_i.shape[0]==level_vox_i.shape[1]==level_vox_i.shape[2])
    S = level_vox_i.shape[0]
    range_min, range_max = opt.eval.range
    # marching cubes treat every cube as unit length
    vertices = vertices/S*(range_max-range_min)+range_min
    mesh = trimesh.Trimesh(vertices, faces)
    meshes[i] = mesh
    if pointclouds is not None:
        # randomly sample on mesh to get uniform dense point cloud
        if len(mesh.triangles)!=0:
            points = mesh.sample(opt.eval.num_points)
        else: points = np.zeros([opt.eval.num_points, 3])
        pointclouds[i] = points

def chamfer_distance(opt, X1, X2):
    assert(X1.shape[2]==3)
    Chamfer_3D = chamfer_3DDist().to(X1.device)
    dist_1, dist_2, idx_1, idx_2 = Chamfer_3D(X1, X2)
    return dist_1.sqrt(), dist_2.sqrt(), idx_1, idx_2

def ICP(opt, X1, X2, num_iter=50): # [B, N, 3]
    assert(len(X1)==len(X2))
    for it in range(num_iter):
        d1, d2, idx, _ = chamfer_distance(opt, X1, X2)
        X2_corresp = torch.zeros_like(X1)
        for i in range(len(X1)):
            X2_corresp[i] = X2[i][idx[i].long()]
        t1 = X1.mean(dim=-2, keepdim=True)
        t2 = X2_corresp.mean(dim=-2, keepdim=True)
        U, S, V = ((X1-t1).transpose(1, 2)@(X2_corresp-t2)).svd(some=True)
        R = V@U.transpose(1, 2)
        R[R.det()<0, 2] *= -1
        X1 = (X1-t1)@R.transpose(1, 2)+t2
    return X1
