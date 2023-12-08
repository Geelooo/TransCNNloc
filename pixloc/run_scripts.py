import pickle


from pixloc.settings import DATA_PATH, LOC_PATH

from pixloc import set_logging_debug, logger
from pixloc.localization import RetrievalLocalizer, PoseLocalizer
from pixloc.utils.data import Paths, create_argparser, parse_paths, parse_conf
from pixloc.utils.io import write_pose_results
from pixloc.utils.eval import cumulative_recall, evaluate
from pixloc.utils.lys_eval import lysevaluate
from pathlib import Path

from pixloc.localization.localizer import lysLocalizer

# 默认路径
default_paths = Paths(
    query_images='{scene}',
    reference_images='{scene}',
    reference_sfm='{scene}_outputs/colmap_db/sfm_superpoint+superglue/',
    query_list='{scene}_outputs/query_list_with_intrinsics.txt',
    retrieval_pairs='{scene}_outputs/netvlad/pairs-query-netvlad.txt',
    ground_truth='{scene}_outputs/colmap_full/sfm_superpoint+superglue/',
    results='{scene}_outputs/results_2023_unet_without_ucm42/pixloc_outputs.txt',
    global_descriptors='{scene}_outputs/colmap_db/global-feats-netvlad.h5'
)

experiment = 'pixloc_cmu'

# 默认配置
default_confs = {
    'from_retrieval': {
        'experiment': experiment,
        'features': {},
        'optimizer': {
            'num_iters': 100,
            'pad': 2,  # to 1?
        },
        'refinement': {
            'num_dbs': 1,
            'multiscale': [2],
            'point_selection': 'all',
            'normalize_descriptors': True,
            'average_observations': False,
            'filter_covisibility': False,
            'do_pose_approximation': False,
        },
    },
    'from_poses': {
        'experiment': experiment,
        'features': {},
        'optimizer': {
            'num_iters': 100,
            'pad': 2,
        },
        'refinement': {
            'num_dbs': 5,
            'min_points_opt': 100,
            'point_selection': 'inliers',
            'normalize_descriptors': True,
            'average_observations': True,
            'layer_indices': [0, 1],
        },
    },
}


def main():
    parser = create_argparser('')
    parser.add_argument('--scene',  nargs='+')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_intermixed_args()

    set_logging_debug(args.verbose)
    paths = parse_paths(args, default_paths)
    conf = parse_conf(args, default_confs)

    # 定位过程
    all_poses = {}
    for scene in args.scene:
        logger.info('Working on scene %s.', scene)
        paths_scene = paths.interpolate(scene=scene)

        # 结果已经输出 利用参数--eval_only跳过再次定位 直接观察结果
        if args.eval_only and paths_scene.results.exists():
            all_poses[scene] = paths_scene.results
            continue

        # 模型的导入
        if args.from_poses:
            localizer = PoseLocalizer(paths_scene, conf)
        else:
            # localizer = RetrievalLocalizer(paths_scene, conf)
            localizer = lysLocalizer(paths_scene, conf)
        # 定位入口
        poses, logs = localizer.run_batched(skip=args.skip)

        # 结果输出
        write_pose_results(poses, paths_scene.results,
                           prepend_camera_name=True)
        with open(f'{paths_scene.results}_logs.pkl', 'wb') as f:
            pickle.dump(logs, f)
        all_poses[scene] = poses

    # 精度评定
    ground_truth = paths_scene.ground_truth
    paths_scene = paths.interpolate(scene=args.scene)
    logger.info('Evaluate scene %s: %s', scene, paths_scene.results)
    out = args.scene[0]+'_outputs'
    list_test = DATA_PATH/out/'list_test.txt'
    global_pos = DATA_PATH/out
    lysevaluate(ground_truth, all_poses[scene],
             global_pos,
             list_test,
             only_localized=(args.skip is not None and args.skip > 1))
    # evaluate(ground_truth, all_poses[scene],
    #          list_test,
    #          only_localized=(args.skip is not None and args.skip > 1))


if __name__ == '__main__':
    main()
