import torch.nn as nn
import numpy as np
import torch

TRANS = -1.5

params = {'vit_b16': {'maxpoolz':5, 'maxpoolxy':11, 'maxpoolpadz':2, 'maxpoolpadxy':5,
                        'convz':5, 'convxy':5, 'convsigmaxy':1, 'convsigmaz':2, 'convpadz':2, 'convpadxy':2,
                        'imgbias':0., 'depth_bias':0.3, 'obj_ratio':0.7, 'bg_clr':0.0,
                        'resolution': 224, 'depth':112}}
net = 'vit_b16'

class Grid2Image(nn.Module):
    def __init__(self):
        super().__init__()
        torch.backends.cudnn.benchmark = False

        self.maxpool = nn.MaxPool3d((params[net]['maxpoolz'], params[net]['maxpoolxy'], params[net]['maxpoolxy']), 
                                    stride=1, 
                                    padding=(params[net]['maxpoolpadz'], params[net]['maxpoolpadxy'], params[net]['maxpoolpadxy']))
        self.conv = torch.nn.Conv3d(1, 3, kernel_size=(params[net]['convz'], params[net]['convxy'], params[net]['convxy']),
                                    stride=1, padding=(params[net]['convpadz'],params[net]['convpadxy'],params[net]['convpadxy']),
                                    bias=True) 
        kn3d = getGaussianKernel3D(params[net]['convxy'], params[net]['convz'], sigma=params[net]['convsigmaxy'], zsigma=params[net]['convsigmaz'])
        self.conv.weight.data = torch.Tensor(kn3d).repeat(3,1,1,1,1)
        self.conv.bias.data.fill_(0.)
            
    def forward(self, x, nnbatch, zz_int, yy, xx):
        x = self.maxpool(x.unsqueeze(1)) 
        x = self.conv(x)

        img = torch.max(x, dim=2)[0]

        temp_max = torch.max(torch.max(img, dim=-1)[0], dim=-1)[0]
        img = img / temp_max[:,:,None,None]
        img = 1 - img

        grid = x / temp_max[:,:,None,None,None]
        grid = 1 - grid
        zz_int = torch.clip(zz_int, 1, params[net]['depth'] - 3)
        point_depth = torch.zeros_like(nnbatch).cuda()
        point_depth = grid[nnbatch.long(), torch.Tensor([0]*nnbatch.shape[0]).cuda().long(), zz_int.view(-1,).long(), yy.view(-1,).long(), xx.view(-1,).long()]
        point_depth = point_depth.view(-1, 2048)
        return img, point_depth

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False
    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach()*0
    one = zero.detach()+1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)
    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat


def points2grid(points, image_height, image_width, depth=params[net]['depth'], device='cpu'):
    batch, pnum, _ = points.shape

    pmax, pmin = points.max(dim=1)[0], points.min(dim=1)[0]
    pcent = (pmax + pmin) / 2
    pcent = pcent[:, None, :]
    prange = (pmax - pmin).max(dim=-1)[0][:, None, None]
    points = (points - pcent) / prange * 2.   # [-1, 1]
    points[:, :, :2] = points[:, :, :2] * params[net]['obj_ratio']  # x, y: [-0.7, 0.7]

    depth_bias = params[net]['depth_bias']
    _x = (points[:, :, 0] + 1) / 2 * image_height  # x: [0.15, 0.85] * image_height
    _y = (points[:, :, 1] + 1) / 2 * image_width   # y: [0.15, 0.85] * image_height
    _z = ((points[:, :, 2] + 1) / 2 + depth_bias) / (1 + depth_bias) * (depth - 2)  # z: [0.2, 1.2] / 1.2 * depth

    # record the coordinate of each point
    _x.ceil_()  # (batch_size * view_num) * 1024 * 1
    _y.ceil_()  # (batch_size * view_num) * 1024 * 1
    z_int = _z.ceil()  # (batch_size * view_num) * 1024 * 1

    _x = torch.clip(_x, 1, params[net]['resolution'] - 2)
    _y = torch.clip(_y, 1, params[net]['resolution'] - 2)
    _z = torch.clip(_z, 1, depth - 2) 

    # nbatch: [0,0,0...0,0,0, 1,1,1...1,1,1, 2,2,2,...2,2,2, ... 14,14,14, 15,15,15]
    nbatch = torch.repeat_interleave(torch.arange(0, batch)[:,None],pnum).view(-1, ).to(device)
    coordinates = torch.cat((nbatch, z_int.view(-1), _y.view(-1), _x.view(-1)), dim=0).view(-1, ).long()
    index = torch.chunk(coordinates, 4, dim=0)
    
    # Grid: (batch_size * view_num) * 112 * 224 *224
    grid = torch.ones([batch, depth, image_height, image_width], device=points.device) * params[net]['bg_clr']  
    grid = grid.index_put(index, _z.view(-1,)).permute((0,1,3,2))
    return grid.squeeze(), _x, _y, z_int, _z, nbatch


class Realistic_Projection:
    """For creating images from PC based on the view information.
    """
    def __init__(self, gpu='cuda:0'):
        _views = np.asarray([
            [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
            ])
        
        # adding some bias to the view angle to reveal more surface
        _views2 = np.asarray([
            [[np.pi / 4, 0, 0], [-0.5, 0, TRANS]],
            [[np.pi / 4, 0, 0], [-0.5, 0, TRANS]],
            [[np.pi / 4, 0, 0], [-0.5, 0, TRANS]],
            [[np.pi / 4, 0, 0], [-0.5, 0, TRANS]],
            [[0, 0, 0], [-0.5, 0, TRANS]],
            [[0, 0, 0], [-0.5, 0, TRANS]],
            [[0, 0, 0], [-0.5, 0, TRANS]],
            [[0, 0, 0], [-0.5, 0, TRANS]],
            [[np.pi / 15, 0, 0], [-0.5, 0, TRANS]],
            [[np.pi / 15, 0, 0], [-0.5, 0, TRANS]],
            ])
        _views3 = np.asarray([
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
             [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]],
             [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, TRANS]],
            ])

        self.num_views = 10
        self.device = torch.device(gpu)

        angle = torch.tensor(_views[:, 0, :]).float().to(self.device)
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        angle2 = torch.tensor(_views2[:, 0, :]).float().to(self.device)
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)
        angle3 = torch.tensor(_views3[:, 0, :]).float().to(self.device)
        self.rot_mat3 = euler2mat(angle3).transpose(1, 2)

        self.translation = torch.tensor(_views[:, 1, :]).float().to(self.device)
        self.translation = self.translation.unsqueeze(1)

        self.grid2image = Grid2Image().to(self.device)

    def get_img(self, points):
        """Get images from point cloud.
        Args:
            points (torch.tensor): of size [B, num_points, 3]
        Returns:
            img (torch.tensor): of size [B * self.num_views, resolution, resolution]
            is_seen (torch.tensor, bool): of size [B * self.num_views, num_points, 1], if the point can be seen in each view
            point_loc_in_img (torch.tensor): of size [B * self.num_views, num_points, 2], point location in each view
        """
        b, _, _ = points.shape
        v = self.translation.shape[0]

        self._points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1),
            rot_mat3=self.rot_mat3.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))
            
        grid, xx, yy, zz_int, zz, nnbatch = points2grid(points=self._points, image_height=params[net]['resolution'], image_width=params[net]['resolution'], device=self.device)        
        img, pc_depth = self.grid2image(grid, nnbatch, zz_int, yy, xx)

        is_seen, point_loc_in_img = self.tell_seen_unseen(img, pc_depth, nnbatch, zz_int, xx, yy)
        return img, is_seen, point_loc_in_img

    def tell_seen_unseen(self, img, pc_depth, nnbatch, zz_int, xx, yy):
        """To determine whether each point can be seen in each view angle, and its location.
        Args:
            points (torch.tensor): of size [B, num_points, 3]
        Returns:
            is_seen (torch.tensor, bool): of size [B * self.num_views, num_points, 1], if the point can be seen in each view
            point_loc_in_img (torch.tensor): of size [B * self.num_views, num_points, 2], point location in each view
        """
        batch, pnum = self._points.shape[0], self._points.shape[1]

        zz_int = torch.clip(zz_int, 1, params[net]['depth'] - 3)
        
        pc_depth_from_img = img[nnbatch.long(), torch.Tensor([0]*(pnum*batch)).view(-1,).long(), yy.view(-1,).long(), xx.view(-1,).long()]
        pc_depth_from_img = pc_depth_from_img.view(-1, pnum)

        unseen_mark = torch.zeros_like(pc_depth_from_img)
        seen_mark = torch.ones_like(pc_depth_from_img)

        is_seen = torch.where(torch.abs(pc_depth_from_img - pc_depth) < 0.1, seen_mark, unseen_mark)

        point_loc_in_img = torch.cat([yy.view(-1,)[:,None], xx.view(-1,)[:,None]], dim=1).view(-1, pnum, 2)
        return is_seen, point_loc_in_img

    @staticmethod
    def point_transform(points, rot_mat, rot_mat2, rot_mat3, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        points = torch.matmul(points, rot_mat3)
        points = points - translation
        return points

def getGaussianKernel2D(ksize, sigma=0):
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kn = kernel1d[..., None] @ kernel1d[None, ...] 
    kn = kn / kn.sum()
    return kn

def getGaussianKernel3D(ksize, depth, sigma=2, zsigma=2):
    k2d = getGaussianKernel2D(ksize, sigma)
    zs = (np.arange(depth, dtype=np.float32) - depth//2)
    zkernel = np.exp(-(zs ** 2) / (2 * zsigma ** 2))
    k3d = np.repeat(k2d[None,:,:], depth, axis=0) * zkernel[:,None, None]
    k3d = k3d / np.sum(k3d)
    k3d = k3d[None, None, :, :, :]
    return k3d 
 

    