import numpy as np
import os
import torch
import torchvision
import torchvision.transforms.functional as torchvision_F
import matplotlib.pyplot as plt
import PIL
import PIL.ImageDraw
from PIL import Image, ImageFont
import trimesh
import pyrender
import cv2
import copy
import base64
import io
import imageio

os.environ['PYOPENGL_PLATFORM'] = 'egl'
@torch.no_grad()
def tb_image(opt, tb, step, group, name, images, masks=None, num_vis=None, from_range=(0, 1), poses=None, cmap="gray", depth=False):
    if not depth:
        images = preprocess_vis_image(opt, images, masks=masks, from_range=from_range, cmap=cmap) # [B, 3, H, W]
    else:
        masks = (masks > 0.5).float()
        images = images * masks + (1 - masks) * ((images * masks).max())
        images = (1 - images).detach().cpu()
    num_H, num_W = num_vis or opt.tb.num_images
    images = images[:num_H*num_W]
    if poses is not None:
        # poses: [B, 3, 4]
        # rots: [max(B, num_images), 3, 3]
        rots = poses[:num_H*num_W, ..., :3]
        images = torch.stack([draw_pose(opt, image, rot, size=20, width=2) for image, rot in zip(images, rots)], dim=0)
    image_grid = torchvision.utils.make_grid(images[:, :3], nrow=num_W, pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:, 3:], nrow=num_W, pad_value=1.)[:1]
        image_grid = torch.cat([image_grid, mask_grid], dim=0)
    tag = "{0}/{1}".format(group, name)
    tb.add_image(tag, image_grid, step)

def preprocess_vis_image(opt, images, masks=None, from_range=(0, 1), cmap="gray"):
    min, max = from_range
    images = (images-min)/(max-min)
    if masks is not None:
        # then the mask is directly the transparency channel of png
        images = torch.cat([images, masks], dim=1)
    images = images.clamp(min=0, max=1).cpu()
    if images.shape[1]==1:
        images = get_heatmap(opt, images[:, 0].cpu(), cmap=cmap)
    return images

def preprocess_depth_image(opt, depth, mask=None, max_depth=1000):
    if mask is not None: depth = depth * mask + (1 - mask) * max_depth  # min of this will leads to minimum of masked regions
    depth = depth - depth.min()
    
    if mask is not None: depth = depth * mask   # max of this will leads to maximum of masked regions
    depth = depth / depth.max()
    return depth

def dump_images(opt, idx, name, images, masks=None, from_range=(0, 1), poses=None, metrics=None, cmap="gray", folder='dump'):
    images = preprocess_vis_image(opt, images, masks=masks, from_range=from_range, cmap=cmap) # [B, 3, H, W]
    if poses is not None:
        rots = poses[..., :3]
        images = torch.stack([draw_pose(opt, image, rot, size=20, width=2) for image, rot in zip(images, rots)], dim=0)
    if metrics is not None:
        images = torch.stack([draw_metric(opt, image, metric.item()) for image, metric in zip(images, metrics)], dim=0)
    images = images.cpu().permute(0, 2, 3, 1).contiguous().numpy() # [B, H, W, 3]
    for i, img in zip(idx, images):
        fname = "{}/{}/{}_{}.png".format(opt.output_path, folder, i, name)
        img = Image.fromarray((img*255).astype(np.uint8))
        img.save(fname)

def dump_depths(opt, idx, name, depths, masks=None, rescale=False, folder='dump'):
    if rescale:
        masks = (masks > 0.5).float()
        depths = depths * masks + (1 - masks) * ((depths * masks).max())
    depths = (1 - depths).detach().cpu()
    for i, depth in zip(idx, depths):
        fname = "{}/{}/{}_{}.png".format(opt.output_path, folder, i, name)
        plt.imsave(fname, depth.squeeze(), cmap='viridis')

# img_list is a list of length n_views, where each view is a image tensor of [B, 3, H, W] 
def dump_gifs(opt, idx, name, imgs_list, from_range=(0, 1), folder='dump', cmap="gray"):
    for i in range(len(imgs_list)):
        imgs_list[i] = preprocess_vis_image(opt, imgs_list[i], from_range=from_range, cmap=cmap)
    for i in range(len(idx)):
        img_list_np = [imgs[i].cpu().permute(1, 2, 0).contiguous().numpy() for imgs in imgs_list]  # list of [H, W, 3], each item is a view of ith sample
        img_list_pil = [Image.fromarray((img*255).astype(np.uint8)).convert('RGB') for img in img_list_np]
        fname = "{}/{}/{}_{}.gif".format(opt.output_path, folder, idx[i], name)
        img_list_pil[0].save(fname, format='GIF', append_images=img_list_pil[1:], save_all=True, duration=100, loop=0)

# img_list is a list of length n_views, where each view is a image tensor of [B, 3, H, W] 
def dump_attentions(opt, idx, name, attn_vis, folder='dump'):
    for i in range(len(idx)):
        img_list_pil = [Image.fromarray((img*255).astype(np.uint8)).convert('RGB') for img in attn_vis[i]]
        fname = "{}/{}/{}_{}.gif".format(opt.output_path, folder, idx[i], name)
        img_list_pil[0].save(fname, format='GIF', append_images=img_list_pil[1:], save_all=True, duration=50, loop=0)

def get_heatmap(opt, gray, cmap): # [N, H, W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[..., :3]).permute(0, 3, 1, 2).contiguous().float() # [N, 3, H, W]
    return color

def dump_meshes(opt, idx, name, meshes, folder='dump'):
    for i, mesh in zip(idx, meshes):
        fname = "{}/{}/{}_{}.ply".format(opt.output_path, folder, i, name)
        try:
            mesh.export(fname)
        except:
            print('Mesh is empty!')

def dump_meshes_viz(opt, idx, name, meshes, save_frames=True, folder='dump'):
    for i, mesh in zip(idx, meshes):
        mesh = copy.deepcopy(mesh)
        R = trimesh.transformations.rotation_matrix(np.radians(180), [0,0,1])
        mesh.apply_transform(R)
        R = trimesh.transformations.rotation_matrix(np.radians(180), [0,1,0])
        mesh.apply_transform(R)
        # our  marching cubes outputs inverted normals for some reason so this is necessary
        trimesh.repair.fix_inversion(mesh) 

        fname = "{}/{}/{}_{}".format(opt.output_path, folder, i, name)
        try:
            mesh = scale_to_unit_cube(mesh)
            visualize_mesh(mesh, fname, write_frames=save_frames)
        except:
            pass

def dump_seen_surface(opt, idx, obj_name, img_name, seen_projs, folder='dump'):
    # seen_proj: [B, H, W, 3]
    for i, seen_proj in zip(idx, seen_projs):
        out_folder = "{}/{}".format(opt.output_path, folder)
        img_fname = "{}_{}.png".format(i, img_name)
        create_seen_surface(i, img_fname, seen_proj, out_folder, obj_name)

# https://github.com/princeton-vl/oasis/blob/master/utils/vis_mesh.py
def create_seen_surface(sample_ID, img_path, XYZ, output_folder, obj_name, connect_thres=0.005):
    height, width = XYZ.shape[:2]
    XYZ_to_idx = {}
    idx = 1
    with open("{}/{}_{}.mtl".format(output_folder, sample_ID, obj_name), "w") as f:
        f.write("newmtl material_0\n")
        f.write("Ka 0.200000 0.200000 0.200000\n")
        f.write("Kd 0.752941 0.752941 0.752941\n")
        f.write("Ks 1.000000 1.000000 1.000000\n")
        f.write("Tr 1.000000\n")
        f.write("illum 2\n")
        f.write("Ns 0.000000\n")
        f.write("map_Ka %s\n" % img_path)
        f.write("map_Kd %s\n" % img_path)

    with open("{}/{}_{}.obj".format(output_folder, sample_ID, obj_name), "w") as f:
        f.write("mtllib {}_{}.mtl\n".format(sample_ID, obj_name))
        for y in range(height):
            for x in range(width):
                if XYZ[y][x][2] > 0:
                    XYZ_to_idx[(y, x)] = idx
                    idx += 1
                    f.write("v %.4f %.4f %.4f\n" % (XYZ[y][x][0], XYZ[y][x][1], XYZ[y][x][2]))
                    f.write("vt %.8f %.8f\n" % ( float(x) / float(width), 1.0 - float(y) / float(height)))
        f.write("usemtl material_0\n")
        for y in range(height-1):
            for x in range(width-1):
                if XYZ[y][x][2] > 0 and XYZ[y][x+1][2] > 0 and XYZ[y+1][x][2] > 0:
                    # if close enough, connect vertices to form a face
                    if torch.norm(XYZ[y][x] - XYZ[y][x+1]).item() < connect_thres and torch.norm(XYZ[y][x] - XYZ[y+1][x]).item() < connect_thres:
                        f.write("f %d/%d %d/%d %d/%d\n" % (XYZ_to_idx[(y, x)], XYZ_to_idx[(y, x)], XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y+1, x)], XYZ_to_idx[(y+1, x)]))
                if XYZ[y][x+1][2] > 0 and XYZ[y+1][x+1][2] > 0 and XYZ[y+1][x][2] > 0:
                    if torch.norm(XYZ[y][x+1] - XYZ[y+1][x+1]).item() < connect_thres and torch.norm(XYZ[y][x+1] - XYZ[y+1][x]).item() < connect_thres:
                        f.write("f %d/%d %d/%d %d/%d\n" % (XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y, x+1)], XYZ_to_idx[(y+1, x+1)], XYZ_to_idx[(y+1, x+1)], XYZ_to_idx[(y+1, x)], XYZ_to_idx[(y+1, x)]))

def dump_pointclouds_compare(opt, idx, name, preds, gts, folder='dump'):
    for i in range(len(idx)):
        pred = preds[i].cpu().numpy()   # [N1, 3]
        gt = gts[i].cpu().numpy()   # [N2, 3]
        color_pred = np.zeros(pred.shape).astype(np.uint8)
        color_pred[:, 0] = 255
        color_gt = np.zeros(gt.shape).astype(np.uint8)
        color_gt[:, 1] = 255
        pc_vertices = np.vstack([pred, gt])
        colors = np.vstack([color_pred, color_gt])
        pc_color = trimesh.points.PointCloud(vertices=pc_vertices, colors=colors)
        fname = "{}/{}/{}_{}.ply".format(opt.output_path, folder, idx[i], name)
        pc_color.export(fname)

def dump_pointclouds(opt, idx, name, pcs, colors, folder='dump', colormap='jet'):
    for i, pc, color in zip(idx, pcs, colors):
        pc = pc.cpu().numpy()   # [N, 3]
        color = color.cpu().numpy()   # [N, 3] or [N, 1]
        # convert scalar color to rgb with colormap
        if color.shape[1] == 1:
            # single channel color in numpy between [0, 1] to rgb
            color = plt.get_cmap(colormap)(color[:, 0])
            color = (color * 255).astype(np.uint8)
        pc_color = trimesh.points.PointCloud(vertices=pc, colors=color)
        fname = "{}/{}/{}_{}.ply".format(opt.output_path, folder, i, name)
        pc_color.export(fname)

@torch.no_grad()
def vis_pointcloud(opt, vis, step, split, pred, GT=None):
    win_name = "{0}/{1}".format(opt.group, opt.name)
    pred, GT = pred.cpu().numpy(), GT.cpu().numpy()
    for i in range(opt.visdom.num_samples):
        # prediction
        data = [dict(
            type="scatter3d",
            x=[float(n) for n in points[i, :opt.visdom.num_points, 0]],
            y=[float(n) for n in points[i, :opt.visdom.num_points, 1]],
            z=[float(n) for n in points[i, :opt.visdom.num_points, 2]],
            mode="markers",
            marker=dict(
                color=color,
                size=1,
            ),
        ) for points, color in zip([pred, GT], ["blue", "magenta"])]
        vis._send(dict(
            data=data,
            win="{0} #{1}".format(split, i),
            eid="{0}/{1}".format(opt.group, opt.name),
            layout=dict(
                title="{0} #{1} ({2})".format(split, i, step),
                autosize=True,
                margin=dict(l=30, r=30, b=30, t=30, ),
                showlegend=False,
                yaxis=dict(
                    scaleanchor="x",
                    scaleratio=1,
                )
            ),
            opts=dict(title="{0} #{1} ({2})".format(win_name, i, step), ),
        ))

@torch.no_grad()
def draw_pose(opt, image, rot_mtrx, size=15, width=1):
    # rot_mtrx: [3, 4]
    mode = "RGBA" if image.shape[0]==4 else "RGB"
    image_pil = torchvision_F.to_pil_image(image.cpu()).convert("RGBA")
    draw_pil = PIL.Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(draw_pil)
    center = (size, size)
    # first column of rotation matrix is the rotated vector of [1, 0, 0]'
    # second column of rotation matrix is the rotated vector of [0, 1, 0]'
    # third column of rotation matrix is the rotated vector of [0, 0, 1]'
    # then always take the first two element of each column is a projection to the 2D plane for visualization
    endpoint = [(size+size*p[0], size+size*p[1]) for p in rot_mtrx.t()]
    draw.line([center, endpoint[0]], fill=(255, 0, 0), width=width)
    draw.line([center, endpoint[1]], fill=(0, 255, 0), width=width)
    draw.line([center, endpoint[2]], fill=(0, 0, 255), width=width)
    image_pil.alpha_composite(draw_pil)
    image_drawn = torchvision_F.to_tensor(image_pil.convert(mode))
    return image_drawn

@torch.no_grad()
def draw_metric(opt, image, metric):
    mode = "RGBA" if image.shape[0]==4 else "RGB"
    image_pil = torchvision_F.to_pil_image(image.cpu()).convert("RGBA")
    draw_pil = PIL.Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(draw_pil)
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    position = (image_pil.size[0] - 80, image_pil.size[1] - 35)
    draw.text(position, '{:.3f}'.format(metric), fill="red", font=font) 
    image_pil.alpha_composite(draw_pil)
    image_drawn = torchvision_F.to_tensor(image_pil.convert(mode))
    return image_drawn

@torch.no_grad()
def show_att_on_image(img, mask):
    """
    Convert the grayscale attention into heatmap on the image.
    Parameters
    ----------
    img: np.array, [H, W, 3]
        Original colored image in [0, 1].
    mask: np.array, [H, W]
        Attention map in [0, 1].
    Returns
    ----------
    np image with attention applied.
    """
    # check the validity
    assert np.max(img) <= 1
    assert np.max(mask) <= 1
    
    # generate heatmap and normalize into [0, 1]
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # add heatmap onto the image
    merged = heatmap + np.float32(img)

    # re-scale the image
    merged = merged / np.max(merged)
    return merged

def look_at(camera_position, camera_target, up_vector):
	vector = camera_position - camera_target
	vector = vector / np.linalg.norm(vector)

	vector2 = np.cross(up_vector, vector)
	vector2 = vector2 / np.linalg.norm(vector2)

	vector3 = np.cross(vector, vector2)
	return np.array([
		[vector2[0], vector3[0], vector[0], 0.0],
		[vector2[1], vector3[1], vector[1], 0.0],
		[vector2[2], vector3[2], vector[2], 0.0],
		[-np.dot(vector2, camera_position), -np.dot(vector3, camera_position), np.dot(vector, camera_position), 1.0]
	])

def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)
    vertices *= 0.5
	
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def get_positions_and_rotations(n_frames=180, r=1.5): 
    '''
        n_frames: how many frames
        r: how far should the camera be
    '''
    # test case 1
    n_frame_full_circ = n_frames // 3 # frames for a full circle
    n_frame_half_circ = n_frames // 6 # frames for a half circle
    
    # full circle in horizontal axes going from 1 to -1 height axis
    pos1 = [np.array([r*np.cos(theta), elev, r*np.sin(theta)]) 
        for theta, elev in zip(np.linspace(0.5*np.pi,2.5*np.pi, n_frame_full_circ), np.linspace(1,-1,n_frame_full_circ))] 
    # half circle in horizontal axes at fixed -1 height
    pos2 = [np.array([r*np.cos(theta), -1, r*np.sin(theta)]) 
        for theta in np.linspace(2.5*np.pi,3.5*np.pi, n_frame_half_circ)]  
    # full circle in horizontal axes going from -1 to 1 height axis
    pos3 = [np.array([r*np.cos(theta), elev, r*np.sin(theta)]) 
        for theta, elev in zip(np.linspace(3.5*np.pi,5.5*np.pi, n_frame_full_circ), np.linspace(-1,1,n_frame_full_circ))] 
    # half circle in horizontal axes at fixed 1 height 
    pos4 = [np.array([r*np.cos(theta), 1, r*np.sin(theta)]) 
        for theta in np.linspace(3.5*np.pi,4.5*np.pi, n_frame_half_circ)] 

    pos = pos1 + pos2 + pos3 + pos4
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    rot = [look_at(x, target, up) for x in pos]
    return pos, rot

def visualize_mesh(mesh, output_path, resolution=(200,200), write_gif=True, write_frames=True, time_per_frame=80, n_frames=180):
    '''
        mesh: Trimesh mesh object
        output_path: absolute path, ".gif" will get added if write_gif, and this will be used as dirname if write_frames is true
        time_per_frame: how many milliseconds to wait for each frame
        n_frames: how many frames in total
    '''
    
    # set material
    mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.8,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            baseColorFactor=(0.5, 0.5, 0.8, 1.0),
        )  
    # define and add scene elements
    mesh = pyrender.Mesh.from_trimesh(mesh, material=mat)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    light = pyrender.SpotLight(color=np.ones(3), intensity=15.0,
                               innerConeAngle=np.pi/4.0,
                               outerConeAngle=np.pi/4.0)

    scene = pyrender.Scene()
    obj = scene.add(mesh)
    cam = scene.add(camera)
    light = scene.add(light)

    positions, rotations = get_positions_and_rotations(n_frames=n_frames)

    r = pyrender.OffscreenRenderer(*resolution)
    
    # move the camera and generate images
    count = 0
    image_list = []
    for pos, rot in zip(positions, rotations):

        pose = np.eye(4)
        pose[:3, 3] = pos
        pose[:3,:3] = rot[:3,:3]
        
        scene.set_pose(cam, pose)
        scene.set_pose(light, pose)

        color, depth = r.render(scene)
        
        img = Image.fromarray(color, mode="RGB")
        image_list.append(img)
    
    # save to file
    if write_gif:
        image_list[0].save(f"{output_path}.gif", format='GIF', append_images=image_list[1:], save_all=True, duration=80, loop=0)

    if write_frames:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i, img in enumerate(image_list):
            img.save(os.path.join(output_path, f"{i:04d}.jpg"))

def get_base64_encoded_image(image_path):
    """
    Returns the base64-encoded image at the given path.
    
    Args:
    image_path (str): The path to the image file.
    
    Returns:
    str: The base64-encoded image.
    """
    with open(image_path, "rb") as f:
        img = Image.open(f)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # Resize the image to reduce its file size
        img.thumbnail((200, 200))
        buffer = io.BytesIO()
        # Convert the image to JPEG format to reduce its file size
        img.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_base64_encoded_gif(gif_path):
    """
    Returns the base64-encoded GIF at the given path.
    
    Args:
    gif_path (str): The path to the GIF file.
    
    Returns:
    str: The base64-encoded GIF.
    """
    with open(gif_path, "rb") as f:
        frames = imageio.mimread(f)
        # Reduce the number of frames to reduce the file size
        frames = frames[::4]
        buffer = io.BytesIO()
        # compress each image frame to reduce the file size
        frames = [frame[::2, ::2] for frame in frames]
        # Convert the GIF to a subrectangle format to reduce the file size
        imageio.mimsave(buffer, frames, format="GIF", fps=10, subrectangles=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def create_gif_html(folder_path, html_file, skip_every=10):
    """
    Creates an HTML file with a grid of sample visualizations.
    
    Args:
    folder_path (str): The path to the folder containing the sample visualizations.
    html_file (str): The name of the HTML file to create.
    """
    # convert path to absolute path
    folder_path = os.path.abspath(folder_path)
    
    # Get a list of all the sample IDs
    ids = []
    count = 0
    all_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split("_")[0]))
    for filename in all_files:
        if filename.endswith("_image_input.png"):
            if count % skip_every == 0:
                ids.append(filename.split("_")[0])
            count += 1

    # Write the HTML file
    with open(html_file, "w") as f:
        # Write the HTML header and CSS style
        f.write("<html>\n")
        f.write("<head>\n")
        f.write("<style>\n")
        f.write(".sample-container {\n")
        f.write("  display: inline-block;\n")
        f.write("  margin: 10px;\n")
        f.write("  width: 350px;\n")
        f.write("  height: 150px;\n")
        f.write("  text-align: center;\n")
        f.write("}\n")
        f.write(".sample-container:nth-child(6n+1) {\n")
        f.write("  clear: left;\n")
        f.write("}\n")
        f.write(".image-container, .gif-container {\n")
        f.write("  display: inline-block;\n")
        f.write("  margin: 10px;\n")
        f.write("  width: 90px;\n")
        f.write("  height: 90px;\n")
        f.write("  object-fit: cover;\n")
        f.write("}\n")
        f.write("</style>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        
        # Write the sample visualizations to the HTML file
        for sample_id in ids:
            try:
                f.write("<div class=\"sample-container\">\n")
                f.write(f"<div class=\"sample-id\"><p>{sample_id}</p></div>\n")
                f.write(f"<div class=\"image-container\"><img src=\"data:image/png;base64,{get_base64_encoded_image(os.path.join(folder_path, sample_id + '_image_input.png'))}\" width=\"90\" height=\"90\"></div>\n")
                f.write(f"<div class=\"image-container\"><img src=\"data:image/png;base64,{get_base64_encoded_image(os.path.join(folder_path, sample_id + '_depth_est.png'))}\" width=\"90\" height=\"90\"></div>\n")
                f.write(f"<div class=\"gif-container\"><img src=\"data:image/gif;base64,{get_base64_encoded_gif(os.path.join(folder_path, sample_id + '_mesh_viz.gif'))}\" width=\"90\" height=\"90\"></div>\n")
                f.write("</div>\n")
            except:
                pass
        
        # Write the HTML footer
        f.write("</body>\n")
        f.write("</html>\n")