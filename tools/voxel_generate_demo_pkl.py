import pickle
import numpy as np
import os
import mmcv
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description="Sweep data generation.")
parser.add_argument("--split", default = "", type = str, help = "The experiment id of this run.")
args = parser.parse_args()

info_prefix = args.split
assert info_prefix in ["train", "val", "test"]

data_root = "./data/nuscenes/"
info_path = os.path.join(data_root,'voxel_nuscenes_temporal_infos_{}_demo.pkl'.format(info_prefix))
key_infos = pickle.load(open(os.path.join(data_root,'voxel_nuscenes_temporal_infos_{}.pkl'.format(info_prefix)), 'rb'))

# num_samples
num_samples = 200
demo_info = []
demo_infos = dict()
demo_info.extend(key_infos['infos'][:num_samples])
demo_infos['infos'] = demo_info
demo_infos['metadata'] = key_infos['metadata']

mmcv.dump(demo_infos, info_path)