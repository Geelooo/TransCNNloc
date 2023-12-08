from pathlib import Path
from pprint import pformat
import numpy as np

from pixloc.localization.localizer import lysLocalizer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import HTML

from pixloc.settings import DATA_PATH, LOC_PATH
from pixloc.localization import RetrievalLocalizer, SimpleTracker
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.visualization.viz_2d import (
    plot_images, plot_keypoints, add_text, features_to_RGB,plot_keypoints1)
from pixloc.visualization.animation import (
    subsample_steps, VideoWriter, create_viz_dump, display_video)


# ------------------------------------------------------------------------
dataset = 'jiawei_cheku'

if dataset == 'Aachen':
    from pixloc.run_Aachen import default_paths, default_confs
elif dataset == 'CMU':
    from pixloc.run_CMU import default_paths, default_confs
    default_paths = default_paths.interpolate(slice=21)
else:
    from pixloc.run_scripts import default_paths, default_confs
    default_paths = default_paths.interpolate(scene=dataset)
    
print(f'default paths:\n{pformat(default_paths.asdict())}')
paths = default_paths.add_prefixes(DATA_PATH, LOC_PATH)

conf = default_confs['from_retrieval']
conf['refinement']['do_pose_approximation'] = False
print(f'conf:\n{pformat(conf)}')

# localizer = RetrievalLocalizer(paths, conf)
localizer=lysLocalizer(paths, conf)
# ------------------------------------------------------------------------
# 载入某一张图片
# name_q = np.random.choice(list(localizer.queries))  # pick a random query
name_q = 'query/IMG_0001.JPG' # or select one for Aachen

tracker = SimpleTracker(localizer.refiner)  # will hook & store the predictions
cam_q = Camera.from_colmap(localizer.queries[name_q])
ret = localizer.run_query(name_q, cam_q)
# ------------------------------------------------------------------------
# 展示CNN特征提取图和预测图 
ref_ids = ret['dbids'][:3]  # show 3 references at most
names_r = [localizer.model3d.dbs[i].name for i in ref_ids]
for name in names_r + [name_q]:
    image, features, weights = tracker.dense[name][2]
    features = features_to_RGB(features[1].numpy())[0]
    plot_images([image, features, weights[1]], cmaps='turbo',
                titles=[name, 'features', 'confidence'], dpi=50)
# ------------------------------------------------------------------------
ref_id = ref_ids[0]
name_r = names_r[0]
ref = localizer.model3d.dbs[ref_id]
cam_r = Camera.from_colmap(localizer.model3d.cameras[ref.camera_id])
T_w2r = Pose.from_colmap(ref)

image_r = tracker.dense[name_r][2][0]
image_q = tracker.dense[name_q][2][0]
p3d = tracker.p3d
T_w2q_init = tracker.T[0]
T_w2q_final = tracker.T[-1]

# Project the 3D points
p2d_r, mask_r = cam_r.world2image(T_w2r * p3d)
p2d_f, mask_f = cam_q.world2image(T_w2q_final * p3d)
p2d_i, mask_i = cam_q.world2image(T_w2q_init * p3d)

plot_images([image_r, image_q], dpi=75)
plot_keypoints1([p2d_r[mask_r], p2d_f[mask_f]], 'lime')
plot_keypoints1([None, p2d_i[mask_i]], 'red')
add_text(0, 'reference')
add_text(1, 'query')
