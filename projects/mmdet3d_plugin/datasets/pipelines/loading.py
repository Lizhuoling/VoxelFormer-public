import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from einops import rearrange

@PIPELINES.register_module()
class LoadMapsFromFiles(object):
    def __init__(self,k=None):
        self.k=k
    def __call__(self,results):
        map_filename=results['map_filename']
        maps=np.load(map_filename)
        map_mask=maps['arr_0'].astype(np.float32)
        
        maps=map_mask.transpose((2,0,1))
        results['gt_map']=maps
        maps=rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps=maps.reshape(256,3*256)
        results['map_shape']=maps.shape
        results['maps']=maps
        return results

@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, 
                sweeps_num=5,
                to_float32=False, 
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=False,
                sweep_range=[3,27],
                sweeps_id = None,
                color_type='unchanged',
                sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                test_mode=True,
                prob=1.0,
                use_can_bus=False,
                ):

        self.sweeps_num = sweeps_num    
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        self.use_can_bus = use_can_bus
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))

                if self.use_can_bus:
                    results['prev_can_bus'] = None
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results['sweeps']):
                        sweep_range = list(range(self.sweep_range[0], min(self.sweep_range[1], len(results['sweeps']))))
                    else:
                        sweep_range = list(range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1])/2) - 1] 
                
            for idx in choices:
                sweep_idx = min(idx, len(results['sweeps']) - 1)
                sweep = results['sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['sweeps'][sweep_idx - 1]
                results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors], axis=-1)
                
                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6  for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)

                can_bus = []
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    results['extrinsics'].append(sweep[sensor]['extrinsics'])

                    if self.use_can_bus:
                        assert 'can_bus' in sweep[sensor]
                        cam_can_bus = sweep[sensor]['can_bus']
                        rotation = Quaternion(sweep[sensor]['ego2global_rotation'])
                        translation = sweep[sensor]['ego2global_translation']
                        cam_can_bus[:3] = translation
                        cam_can_bus[3:7] = rotation
                        patch_angle = quaternion_yaw(rotation) / np.pi * 180
                        if patch_angle < 0:
                            patch_angle += 360
                        cam_can_bus[-2] = patch_angle / 180 * np.pi
                        cam_can_bus[-1] = patch_angle
                        can_bus.append(cam_can_bus)

                if self.use_can_bus:
                    can_bus = np.array(can_bus).mean(axis=0)
                    results['prev_can_bus'] = can_bus
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_imgs_list

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewImageDepthMap(object):
    def __init__(self, num=6):
        self.num = num

    def __call__(self, results):
        eps = 1e-5

        pts_filename = results['pts_filename']
        points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(-1, 5)[..., :4] #(x, y, z, intensity)
        points = np.concatenate((points[..., :3], np.ones_like(points[..., 3:4])), axis=-1)

        gt_depths = []
        for i in range(self.num):
            img = results['img'][i]
            lidar2img = results['lidar2img'][i]
            img_points  = np.matmul(lidar2img, points[:, :, None])[..., 0]
            img_coords = img_points[..., :3]
            img_coords[..., :2] = img_coords[..., :2] / np.maximum(img_coords[..., 2:3], np.ones_like(img_coords[..., 2:3]) * eps)

            mask = np.ones(img_coords.shape[0], dtype=bool)
            mask = np.logical_and(mask, img_coords[..., 2] > 0.0)
            mask = np.logical_and(mask, img_coords[..., 0] > 0)
            mask = np.logical_and(mask, img_coords[..., 0] < img.shape[1] - 1)
            mask = np.logical_and(mask, img_coords[..., 1] > 0)
            mask = np.logical_and(mask, img_coords[..., 1] < img.shape[0] - 1)

            gt_depth = np.zeros(img.shape[:2])
            gt_depth[img_coords[mask, 1].astype(np.int16), img_coords[mask, 0].astype(np.int16)] = img_coords[mask, 2]
            gt_depths.append(gt_depth)

            '''import matplotlib.pyplot as plt
            plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.imshow(img[:, :, ::-1].astype(np.uint8))
            plt.scatter(img_coords[mask, 0],img_coords[mask, 1],c=img_coords[mask, 2],cmap='rainbow_r',alpha=0.5,s=4)
            plt.savefig('vis_depth.png', bbox_inches='tight')
            plt.imsave('vis_depth_map.png', gt_depth)
            import pdb
            pdb.set_trace()'''

        results['gt_depth'] = gt_depths

        return results

@PIPELINES.register_module()
class LoadCanbus(object):
    def __init__(self):
        pass

    def __call__(self, results):
        assert 'can_bus' in results
        can_bus = results['can_bus']
        rotation = Quaternion(results['ego2global_rotation'])
        translation = results['ego2global_translation']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        if 'prev_can_bus' in results.keys():
            prev_can_bus = results['prev_can_bus']
            if prev_can_bus is None:
                # if no prev 
                can_bus[:3] = 0
                can_bus[-1] = 0
                results['cur_can_bus'] = can_bus
                results['prev_can_bus'] = can_bus
            else:
                can_bus[:3] -= prev_can_bus[:3]
                can_bus[-1] -= prev_can_bus[-1]
                prev_can_bus[:3] = 0
                prev_can_bus[-1] = 0
                results['cur_can_bus'] = can_bus
                results['prev_can_bus'] = prev_can_bus
        else:
             results['prev_can_bus'] = None
        return results