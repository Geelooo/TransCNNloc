import logging
import pickle
import time
from typing import Optional, Dict, Tuple, Union
from omegaconf import DictConfig, OmegaConf as oc
from tqdm import tqdm
import torch
from pathlib import Path

from .model3d import Model3D
from .feature_extractor import FeatureExtractor
from .refiners import PoseRefiner, RetrievalRefiner

from ..utils.data import Paths
from ..utils.io import parse_image_lists, parse_retrieval, load_hdf5
from ..utils.quaternions import rotmat2qvec
from ..pixlib.utils.experiments import load_experiment
from ..pixlib.models import get_model
from ..pixlib.geometry import Camera

# hloc
from hloc import lys_extract_features
from hloc.utils.base_model import dynamic_load
from hloc import extractors
from hloc.utils.io import list_h5_names
from hloc.lys_extract_features import parse_names,get_descriptors


logger = logging.getLogger(__name__)
# TODO: despite torch.no_grad in BaseModel, requires_grad flips in ref interp
torch.set_grad_enabled(False)


class Localizer:
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
        # 模型导入
        start_time=time.time()
        self.model3d = Model3D(paths.reference_sfm)
        cameras = parse_image_lists(paths.query_list, with_intrinsics=True)
 
        self.queries = {n: c for n, c in cameras}

        # Loading feature extractor and optimizer from experiment or scratch
        conf = oc.create(conf)
        conf_features = conf.features.get('conf', {})
        conf_optim = conf.get('optimizer', {})
        if conf.get('experiment'):
            pipeline = load_experiment(
                    conf.experiment,
                    {'extractor': conf_features, 'optimizer': conf_optim})
            pipeline = pipeline.to(device)
            logger.debug(
                'Use full pipeline from experiment %s with config:\n%s',
                conf.experiment, oc.to_yaml(pipeline.conf))
            extractor = pipeline.extractor
            optimizer = pipeline.optimizer
            if isinstance(optimizer, torch.nn.ModuleList):
                optimizer = list(optimizer)
        else:
            assert 'name' in conf.features
            extractor = get_model(conf.features.name)(conf_features)
            optimizer = get_model(conf.optimizer.name)(conf_optim)

        self.paths = paths
        self.conf = conf
        self.device = device
        self.optimizer = optimizer
        self.extractor = FeatureExtractor(
            extractor, device, conf.features.get('preprocessing', {}))

    def run_query(self, name: str, camera: Camera):
        raise NotImplementedError

    def run_batched(self, skip: Optional[int] = None,
                    ) -> Tuple[Dict[str, Tuple], Dict]:
        output_poses = {}
        output_logs = {
            'paths': self.paths.asdict(),
            'configuration': oc.to_yaml(self.conf),
            'localization': {},
        }

        logger.info('Starting the localization process...')
        query_names = list(self.queries.keys())[::skip or 1]
        for name in tqdm(query_names):
            camera = Camera.from_colmap(self.queries[name])
            try:
                ret = self.run_query(name, camera)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    logger.info('Out of memory')
                    torch.cuda.empty_cache()
                    ret = {'success': False}
                else:
                    raise
            output_logs['localization'][name] = ret
            if ret['success']:
                R, tvec = ret['T_refined'].numpy()
            elif 'T_init' in ret:
                R, tvec = ret['T_init'].numpy()
            else:
                continue
            output_poses[name] = (rotmat2qvec(R), tvec)

        return output_poses, output_logs


class RetrievalLocalizer(Localizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(paths, conf, device)

        # if paths.global_descriptors is not None:
        #     global_descriptors = load_hdf5(paths.global_descriptors)
        # else:
        global_descriptors = None

        self.refiner = RetrievalRefiner(
            self.device, self.optimizer, self.model3d, self.extractor, paths,
            self.conf.refinement, global_descriptors=global_descriptors)

        if paths.hloc_logs is not None:
            logger.info('Reading hloc logs...')
            with open(paths.hloc_logs, 'rb') as f:
                self.logs = pickle.load(f)['loc']
            self.retrieval = {q: [self.model3d.dbs[i].name for i in loc['db']]
                              for q, loc in self.logs.items()}
        elif paths.retrieval_pairs is not None:
            self.logs = None
            self.retrieval = parse_retrieval(paths.retrieval_pairs)
        else:
            raise ValueError

    # @profile
    def run_query(self, name: str, camera: Camera):
        dbs = [self.model3d.name2id[r] for r in self.retrieval[name]]
        loc = None if self.logs is None else self.logs[name]
        ret = self.refiner.refine(name, camera, dbs, loc=loc)
        return ret


class PoseLocalizer(Localizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(paths, conf, device)

        self.refiner = PoseRefiner(
            device, self.optimizer, self.model3d, self.extractor, paths,
            self.conf.refinement)

        logger.info('Reading hloc logs...')
        with open(paths.hloc_logs, 'rb') as f:
            self.logs = pickle.load(f)['loc']

    def run_query(self, name: str, camera: Camera):
        loc = self.logs[name]
        if loc['PnP_ret']['success']:
            ret = self.refiner.refine(name, camera, loc)
        else:
            ret = {'success': False}
        return ret

class lysLocalizer(Localizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(paths, conf, device)

        self.refiner = RetrievalRefiner(
            self.device, self.optimizer, self.model3d, self.extractor, paths,
            self.conf.refinement, global_descriptors=None)

        self.logs = None
        self.retrieval = None
        # 图像检索模型读取
        self.retrieval_conf = lys_extract_features.confs['netvlad']

        self.image_loader=lys_extract_features.imageloader(self.paths.query_images, self.retrieval_conf['preprocessing'])
        Model = dynamic_load(extractors, self.retrieval_conf['model']['name'])
        self.netmodel = Model(self.retrieval_conf['model']).eval().to(self.device)

        descriptors=self.paths.global_descriptors
        db_descriptors = [descriptors]
        name2db = {n: i for i, p in enumerate(db_descriptors)
                for n in list_h5_names(p)}
        db_names = list(name2db.keys())
        self.db_names = parse_names('db', None, db_names)

        if len(db_names) == 0:
            raise ValueError('Could not find any database image.')
                                                           
        self.db_desc = get_descriptors(db_names, db_descriptors, name2db)


    def run_query(self, name: str, camera: Camera):
        # 开始匹配
        # self.retrieval = [name0,name1,name2,name3,name4]
        dbs=lys_extract_features.image_retrivel(self,db_prefix="db", query_prefix="query_dif_time",
                                                num_matched=5,query_name=name)
        dbs = [self.model3d.name2id[r] for r in dbs]
        loc = None if self.logs is None else self.logs[name]
        ret = self.refiner.refine(name, camera, dbs, loc=loc)
        return ret
#------------------------------------------------------------------------
class baseLocalizer:
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
        # 模型导入
        self.model3d1 = Model3D("/home/lys/Workplace/datasets/ipin_final_outputs/f1")
        self.model3d2 = Model3D("/home/lys/Workplace/datasets/ipin_final_outputs/f2")
        cameras = parse_image_lists(paths.query_list, with_intrinsics=True)
        self.queries = {n: c for n, c in cameras}

        # Loading feature extractor and optimizer from experiment or scratch
        conf = oc.create(conf)
        conf_features = conf.features.get('conf', {})
        conf_optim = conf.get('optimizer', {})
        if conf.get('experiment'):
            pipeline = load_experiment(
                    conf.experiment,
                    {'extractor': conf_features, 'optimizer': conf_optim})
            pipeline = pipeline.to(device)
            logger.debug(
                'Use full pipeline from experiment %s with config:\n%s',
                conf.experiment, oc.to_yaml(pipeline.conf))
            extractor = pipeline.extractor
            optimizer = pipeline.optimizer
            if isinstance(optimizer, torch.nn.ModuleList):
                optimizer = list(optimizer)
        else:
            assert 'name' in conf.features
            extractor = get_model(conf.features.name)(conf_features)
            optimizer = get_model(conf.optimizer.name)(conf_optim)

        self.paths = paths
        self.conf = conf
        self.device = device
        self.optimizer = optimizer
        self.extractor = FeatureExtractor(
            extractor, device, conf.features.get('preprocessing', {}))

    def run_query(self, name: str, camera: Camera):
        raise NotImplementedError

class IPINLocalizer(baseLocalizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None):
        super().__init__(paths, conf, device)

        self.refiner1 = RetrievalRefiner(
            self.device, self.optimizer, self.model3d1, self.extractor, paths,
            self.conf.refinement, global_descriptors=None)
        self.refiner2 = RetrievalRefiner(
            self.device, self.optimizer, self.model3d2, self.extractor, paths,
            self.conf.refinement, global_descriptors=None)

        self.logs = None
        self.retrieval = None
        # 图像检索模型读取
        self.retrieval_conf = lys_extract_features.confs['netvlad']

        self.image_loader=lys_extract_features.imageloader(self.paths.query_images, self.retrieval_conf['preprocessing'])
        Model = dynamic_load(extractors, self.retrieval_conf['model']['name'])
        self.netmodel = Model(self.retrieval_conf['model']).eval().to(self.device)

        descriptors=self.paths.global_descriptors
        db_descriptors = [descriptors]
        name2db = {n: i for i, p in enumerate(db_descriptors)
                for n in list_h5_names(p)}
        db_names = list(name2db.keys())
        self.db_names = parse_names('db', None, db_names)

        if len(db_names) == 0:
            raise ValueError('Could not find any database image.')
                                                           
        self.db_desc = get_descriptors(db_names, db_descriptors, name2db)


    def run_query(self, name: str, camera: Camera):
        # 开始匹配
        # self.retrieval = [name0,name1,name2,name3,name4]
        dbs=lys_extract_features.image_retrivel(self,db_prefix="f", query_prefix="query",
                                                num_matched=5,query_name=name)
        if(dbs[0][1]=="1"):
            print("在一楼")
            dbs = [self.model3d1.name2id[r] for r in dbs]
            loc = None if self.logs is None else self.logs[name]
            ret = self.refiner1.refine(name, camera, dbs, loc=loc)
            return ret
        else:
            print("在二楼")
            dbs = [self.model3d2.name2id[r] for r in dbs]
            loc = None if self.logs is None else self.logs[name]
            ret = self.refiner2.refine(name, camera, dbs, loc=loc)
            return ret
    