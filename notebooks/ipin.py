from pathlib import Path
from pprint import pformat
import numpy as np

from pixloc.localization.localizer import lysLocalizer,IPINLocalizer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import HTML

from pixloc.settings import DATA_PATH, LOC_PATH
from pixloc.localization import RetrievalLocalizer, SimpleTracker
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.utils.quaternions import rotmat2qvec


# ------------------------------------------------------------------------
dataset = 'ipin_final'

from pixloc.run_scripts import default_paths, default_confs
default_paths = default_paths.interpolate(scene=dataset)
    
print(f'default paths:\n{pformat(default_paths.asdict())}')
paths = default_paths.add_prefixes(DATA_PATH, LOC_PATH)

conf = default_confs['from_retrieval']
conf['refinement']['do_pose_approximation'] = False
print(f'conf:\n{pformat(conf)}')

localizer=IPINLocalizer(paths, conf)
# localizer = RetrievalLocalizer(paths, conf)
# ------------------------------------------------------------------------
# 载入某一张图片
# name_q = np.random.choice(list(localizer.queries))  # pick a random query
name_q = 'query/2_2.png' # or select one for Aachen

# tracker = SimpleTracker(localizer.refiner)  # will hook & store the predictions
cam_q = Camera.from_colmap(localizer.queries['query/frame_000007.png'])
ret = localizer.run_query(name_q, cam_q)
if ret['success']:
    R, tvec = ret['T_refined'].numpy()
R=rotmat2qvec(R)

