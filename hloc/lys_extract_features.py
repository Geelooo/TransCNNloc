import argparse
from pickle import NONE

import time
import torch
from pathlib import Path
from typing import Dict, List, Union, Optional
import h5py
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import collections.abc as collections
import PIL.Image

from . import extractors, logger
from .utils.base_model import dynamic_load
from .utils.tools import map_tensor
from .utils.parsers import parse_image_lists
from .utils.io import read_image, list_h5_names


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
'''
confs = {
    'superpoint_aachen': {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    'superpoint_max': {
        'output': 'feats-superpoint-n4096-rmax1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },
    'superpoint_inloc': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'r2d2': {
        'output': 'feats-r2d2-n5000-r1024',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 5000,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'd2net-ss': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'sift': {
        'output': 'feats-sift',
        'model': {
            'name': 'dog'
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'sosnet': {
        'output': 'feats-sosnet',
        'model': {
            'name': 'dog',
            'descriptor': 'sosnet'
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    # Global descriptors
    'dir': {
        'output': 'global-feats-dir',
        'model': {'name': 'dir'},
        'preprocessing': {'resize_max': 1024},
    },
    'netvlad': {
        'output': 'global-feats-netvlad',
        'model': {'name': 'netvlad'},
        'preprocessing': {'resize_max': 1024},
    },
    'openibl': {
        'output': 'global-feats-openibl',
        'model': {'name': 'openibl'},
        'preprocessing': {'resize_max': 1024},
    }
}


def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
    if name2idx is None:
        with h5py.File(str(path), 'r') as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), 'r') as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(scores: torch.Tensor,
                            invalid: np.array,
                            num_select: int,
                            min_score: Optional[float] = None):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float('-inf'))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(f'Unknown type of image list: {names}.'
                             'Provide either a list or a path to a list file.')
    else:
        names = names_all
    return names


def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(PIL.Image, interp[len('pil_'):].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized

class imageloader:
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
        'interpolation': 'cv2_area',  # pil_linear is more accurate but slower
    }

    def __init__(self, root, conf, paths=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

    def get_query_image(self, query):
        image = read_image(self.root / query, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': query,
            'image': image,
            'original_size': np.array(size),
        }
        return data

@torch.no_grad()
def main(conf: Dict,
         image_dir: Path,
         query_prefix=None,db_prefix=None,
         query_name=None,
         num_matched=10) -> Path:
    
    data=self.image_loader.get_query_image(query_name)

    name = data['name']  # remove batch dimension
    data['image']=torch.unsqueeze(torch.tensor(data['image']),0)
    data['original_size']=torch.unsqueeze(torch.tensor(data['original_size']),0)
    pred = self.netmodel(map_tensor(data, lambda x: x.to(self.device)))
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    pred['image_size'] = original_size = data['original_size'][0].numpy()
        
    # 开始匹配
    # -------------------------------------------------
    # descriptors=Path('/home/lys/Workplace/datasets/lys_241_outputs/netvlad/global-feats-netvlad.h5')
    # db_descriptors = [descriptors]
    # name2db = {n: i for i, p in enumerate(db_descriptors)
    #            for n in list_h5_names(p)}
    # db_names = list(name2db.keys())
    # db_names = parse_names('db', None, db_names)

    # if len(db_names) == 0:
    #     raise ValueError('Could not find any database image.')
                                                           
    
    # db_desc = get_descriptors(db_names, db_descriptors, name2db)
    query_names = [name]
    query_desc = torch.unsqueeze(torch.tensor(pred['global_descriptor']),0)
    sim = torch.einsum('id,jd->ij', query_desc.to(self.device), self.db_desc.to(self.device))

    # Avoid self-matching
    self = np.array(query_names)[:, None] == np.array(self.db_names)[None]
    pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
    return [self.db_names[j] for i,j in pairs]
    # -------------------------------------------------


@torch.no_grad()
def image_retrivel( self,
         query_prefix=None,db_prefix=None,
         query_name=None,
         num_matched=10
        ):
    
    data=self.image_loader.get_query_image(query_name)
    #----
    #比赛专用
    
    #----
    name = data['name']  # remove batch dimension
    data['image']=torch.unsqueeze(torch.tensor(data['image']),0)
    data['original_size']=torch.unsqueeze(torch.tensor(data['original_size']),0)
    pred = self.netmodel(map_tensor(data, lambda x: x.to(self.device)))
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    pred['image_size'] = original_size = data['original_size'][0].numpy()
        
    # 开始匹配
    # -------------------------------------------------
    descriptors=self.paths.global_descriptors
    db_descriptors = [descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors)
               for n in list_h5_names(p)}
    db_names = list(name2db.keys())
    db_names = parse_names(db_prefix, None, db_names)

    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')
    query_names = [name]
                                                           
    
    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    # Todo 不用读每一张图片都将所有描述符加载到内存中
    query_desc = torch.unsqueeze(torch.tensor(pred['global_descriptor']),0)
    sim = torch.einsum('id,jd->ij', query_desc.to(self.device), db_desc.to(self.device))

    # Avoid self-matching
    self = np.array(query_names)[:, None] == np.array(db_names)[None]
    start_time = time.time()
    pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
    end_time = time.time()
    execution_time = end_time - start_time
    print("代码执行时间: ", execution_time, "秒")
    return [db_names[j] for i,j in pairs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--query_name', type=str, nargs='+',required=True)
    args = parser.parse_args()
    main(confs[args.conf],**args.__dict__)
