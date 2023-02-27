import torch 
from .array_converter import array_converter
import pdb

@array_converter(apply_to=('points', 'cam2img'))
def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (torch.Tensor): 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img (torch.Tensor): Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].

    Returns:
        torch.Tensor: points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 

    if normalized_bboxes.size(-1) > 8:
        # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

    if normalized_bboxes.size(-1) in [9, 11] : # wth uncertainty
        loc_uncern = normalized_bboxes[:, -1].exp() / 10
        conf_3d = 1 - torch.clamp(loc_uncern, min=0.01, max=0.9)
    else:
        conf_3d = None

    return denormalized_bboxes, conf_3d

def nan_to_num(input, nan = None, posinf = None, neginf = None):
    '''
    Description:
        Since torch1.7 does not have torch.nan_to_num, we implement it by ourselves.
    '''
    if input == None:
        return None

    if input.dtype == torch.float32:
        max_num = 3.4028e+38
    elif input.dtype == torch.float16:
        max_num = 65504
    elif input.dtype == torch.bool:
        max_num = None
    else:
        raise Exception('Not supported input type.')

    mask = input.isnan()
    input[mask] = input[mask].detach()
    if nan == None:
        input[mask] = 0
    else:
        input[mask] = nan

    if input.dtype != torch.bool:
        mask = (input.isinf()) & (input > 0)
        input[mask] = input[mask].detach()
        if posinf == None:
            input[mask] = max_num
        else:
            input[mask] = posinf

        mask = (input.isinf()) & (input < 0)
        input[mask] = input[mask].detach()
        if neginf == None:
            input[mask] = -max_num
        else:
            input[mask] = neginf

    return input