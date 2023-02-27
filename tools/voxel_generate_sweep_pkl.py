
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import pickle
from nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion
import numpy as np
import os
import pdb
import mmcv
import tqdm
import argparse
from copy import deepcopy
sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

parser = argparse.ArgumentParser(description="Sweep data generation.")
parser.add_argument("--split", default = "", type = str, help = "The experiment id of this run.")
args = parser.parse_args()

info_prefix = args.split
assert info_prefix in ["train", "val", "test"]

data_root = "./data/nuscenes/"
num_prev = 5  ###nummber of previous key frames
num_sweep = 5  ###nummber of sweep frames between two key frame

info_path = os.path.join(data_root,'voxel_nuscenes_temporal_infos_{}.pkl'.format(info_prefix))
key_infos = pickle.load(open(os.path.join(data_root,'voxel_nuscenes_infos_{}.pkl'.format(info_prefix)), 'rb'))
if info_prefix == 'test':
    nuscenes_version = 'v1.0-test'
else:
    nuscenes_version = 'v1.0-trainval'
nuscenes = NuScenes(nuscenes_version, data_root)
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
nusc_can_bus = NuScenesCanBus(dataroot=data_root)

def add_frame(sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat, pose_list):
    sweep_cam = dict()
    sweep_cam['is_key_frame'] = sample_data['is_key_frame']
    sweep_cam['data_path'] = os.path.join(data_root, sample_data['filename'])
    sweep_cam['type'] = 'camera'
    sweep_cam['timestamp'] = sample_data['timestamp']
    sweep_cam['sample_data_token'] = sample_data['sample_token']
    pose_record = nuscenes.get('ego_pose', sample_data['ego_pose_token']) ##{'token': '4367ec13cba845aab19cff4973eebc4a', 'timestamp': 1533153862354799, 'rotation': [0.014338564560080185, -0.005652165998640543, 0.023939306730068593, -0.9995946019157788], 'translation': [2365.4560154353267, 796.2968658597514, 0.0]}
    calibrated_sensor_record = nuscenes.get('calibrated_sensor', sample_data['calibrated_sensor_token']) ##{'token': '2fde3d3376ea42a8a561df595e001cc7', 'sensor_token': 'ec4b5d41840a509984f7ec36419d4c09', 'translation': [1.5752559464, 0.500519383135, 1.50696032589], 'rotation': [0.6812088525125634, -0.6687507165046241, 0.2101702448905517, -0.21108161122114324], 'camera_intrinsic': [[1257.8625342125129, 0.0, 827.2410631095686], [0.0, 1257.8625342125129, 450.915498205774], [0.0, 0.0, 1.0]]}

    sample_timestamp = sample_data['timestamp']
    if pose_list is not None:
        can_bus = []
        # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
        temp_pose = pose_list[0]
        for i, pose in enumerate(pose_list):
            if pose['utime'] > sample_timestamp:
                break
            temp_pose = pose
        last_pose = deepcopy(temp_pose)
        _ = last_pose.pop('utime')  # useless
        pos = last_pose.pop('pos')
        rotation = last_pose.pop('orientation')
        can_bus.extend(pos)
        can_bus.extend(rotation)
        for key in last_pose.keys():
            can_bus.extend(pose[key])  # 16 elements
        can_bus.extend([0., 0.])
    else:
        can_bus = np.zeros(18)
    sweep_cam['can_bus'] = can_bus

    sweep_cam['ego2global_translation']  = pose_record['translation']
    sweep_cam['ego2global_rotation']  = pose_record['rotation']
    sweep_cam['sensor2ego_translation']  = calibrated_sensor_record['translation']
    sweep_cam['sensor2ego_rotation']  = calibrated_sensor_record['rotation']
    sweep_cam['cam_intrinsic'] = calibrated_sensor_record['camera_intrinsic']

    l2e_r_s = sweep_cam['sensor2ego_rotation']
    l2e_t_s = sweep_cam['sensor2ego_translation'] 
    e2g_r_s = sweep_cam['ego2global_rotation']
    e2g_t_s = sweep_cam['ego2global_translation'] 

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep_cam['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep_cam['sensor2lidar_translation'] = T

    lidar2cam_r = np.linalg.inv(sweep_cam['sensor2lidar_rotation'])
    lidar2cam_t = sweep_cam['sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.array(sweep_cam['cam_intrinsic'])
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    sweep_cam['intrinsics'] = viewpad.astype(np.float32)
    sweep_cam['extrinsics'] = lidar2cam_rt.astype(np.float32)
    sweep_cam['lidar2img'] = lidar2img_rt.astype(np.float32)

    # pop_keys = ['ego2global_translation', 'ego2global_rotation', 'sensor2ego_translation', 'sensor2ego_rotation', 'cam_intrinsic']
    # keep ego2global for can_bus
    pop_keys = ['sensor2ego_translation', 'sensor2ego_rotation', 'cam_intrinsic']
    [sweep_cam.pop(k) for k in pop_keys]

    return sweep_cam

for current_id in tqdm.tqdm(range(len(key_infos['infos']))):
    ###parameters of current key frame 
    e2g_t = key_infos['infos'][current_id]['ego2global_translation']
    e2g_r = key_infos['infos'][current_id]['ego2global_rotation']
    l2e_t = key_infos['infos'][current_id]['lidar2ego_translation']
    l2e_r = key_infos['infos'][current_id]['lidar2ego_rotation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    sample = nuscenes.get('sample', key_infos['infos'][current_id]['token']) # {'token': 'c0be823ae8f040e2b3306002c571ae57', 'timestamp': 1533153861447131, 'prev': 'e866142822bb421d87d8f9bd1b91fbc3', 'next': 'f32d3a2842004926b41985152fa1bfad', 'scene_token': 'bc6a757d637f4832be68986833ec17ac', 'data': {'RADAR_FRONT': '85962dfd390843bab8cbedc9003a5d81', 'RADAR_FRONT_LEFT': '35e35910a6f8428ea1e3f71db59f0ed7', 'RADAR_FRONT_RIGHT': 'a557a223830d4f7db59a9bf03425c52d', 'RADAR_BACK_LEFT': '46b86e2060e341dabb14396a8edc1653', 'RADAR_BACK_RIGHT': '7e7b5ad41eff4f949d69b3ef6d65f991', 'LIDAR_TOP': '5a0aa6326b004322bf009388f4df33df', 'CAM_FRONT': 'a5c43d3424bd406ba1a0a3d1d1493277', 'CAM_FRONT_RIGHT': '38ee6078f2594c5cb3bea00956d3afeb', 'CAM_BACK_RIGHT': '082193ef4dff4dca9ff7af18493107f5', 'CAM_BACK': 'aec2027af4e243b591cf22459735644e', 'CAM_BACK_LEFT': 'd6c479b792674d8db1a5de86af2b9183', 'CAM_FRONT_LEFT': '451c4acac4534a0da20e652ba49a14a2'}, 'anns': []}
    current_cams = dict() ###cam of current key frame
    for cam in sensors:
        current_cams[cam] = nuscenes.get('sample_data', sample['data'][cam]) ##{'token': '8e25cfcd8f724bb7bbce69bff042a56f', 'sample_token': '02fd302178dd44568ae305320ea24054', 'ego_pose_token': '8e25cfcd8f724bb7bbce69bff042a56f', 'calibrated_sensor_token': '2fde3d3376ea42a8a561df595e001cc7', 'timestamp': 1533153859904816, 'fileformat': 'jpg', 'is_key_frame': True, 'height': 900, 'width': 1600, 'filename': 'samples/CAM_FRONT_LEFT/n008-2018-08-01-16-03-27-0400__CAM_FRONT_LEFT__1533153859904816.jpg', 'prev': '5d82f148ba8947579a6d7647ac73a9d6', 'next': 'cb0a1671873647faba28916a88b14574', 'sensor_modality': 'camera', 'channel': 'CAM_FRONT_LEFT'}
   
    scene_name = nuscenes.get('scene', sample['scene_token'])['name']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        pose_list = None  # server scenes do not have can bus information.
    sweep_lists = []
    for i in range(num_prev):  #### previous sweep frame  
        ### justify the first frame of a scene
        if sample['prev'] == '': 
            break
        ###add sweep frame between two key frame
        for j in range(num_sweep): 
            sweep_cams = dict()
            for cam in sensors: 
                if current_cams[cam]['prev'] == '':    
                    sweep_cams = sweep_lists[-1] 
                    break
                sample_data = nuscenes.get('sample_data', current_cams[cam]['prev']) ##{'token': '8e25cfcd8f724bb7bbce69bff042a56f', 'sample_token': '02fd302178dd44568ae305320ea24054', 'ego_pose_token': '8e25cfcd8f724bb7bbce69bff042a56f', 'calibrated_sensor_token': '2fde3d3376ea42a8a561df595e001cc7', 'timestamp': 1533153859904816, 'fileformat': 'jpg', 'is_key_frame': True, 'height': 900, 'width': 1600, 'filename': 'samples/CAM_FRONT_LEFT/n008-2018-08-01-16-03-27-0400__CAM_FRONT_LEFT__1533153859904816.jpg', 'prev': '5d82f148ba8947579a6d7647ac73a9d6', 'next': 'cb0a1671873647faba28916a88b14574', 'sensor_modality': 'camera', 'channel': 'CAM_FRONT_LEFT'}
                sweep_cam = add_frame(sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat, pose_list)
                current_cams[cam] = sample_data
                sweep_cams[cam] = sweep_cam
            sweep_lists.append(sweep_cams)
        ###add previous key frame
        sample = nuscenes.get('sample', sample['prev'])
        sweep_cams = dict()
        for cam in sensors:
            sample_data = nuscenes.get('sample_data', sample['data'][cam])
            sweep_cam = add_frame(sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat, pose_list)
            current_cams[cam] = sample_data
            sweep_cams[cam] = sweep_cam
        sweep_lists.append(sweep_cams)
    key_infos['infos'][current_id]['sweeps'] = sweep_lists

mmcv.dump(key_infos, info_path)
