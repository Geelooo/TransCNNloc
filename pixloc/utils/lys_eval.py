import logging

from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import numpy as np
from pyrsistent import v
from .io import parse_image_list
from .colmap import qvec2rotmat, read_images_binary, read_images_text

logger = logging.getLogger(__name__)


def lysevaluate(gt_sfm_model: Path, predictions: Union[Dict, Path],
                global_pos: Path,
                test_file_list: Optional[Path] = None,
                only_localized: bool = False):

    if not isinstance(predictions, dict):
        predictions = parse_image_list(predictions, with_poses=True)
        predictions = {n: (im.qvec, im.tvec) for n, im in predictions}

    # ground truth poses from the sfm model
    images_bin = gt_sfm_model / 'images.bin'
    images_txt = gt_sfm_model / 'images.txt'
    if images_bin.exists():
        images = read_images_binary(images_bin)
    elif images_txt.exists():
        images = read_images_text(images_txt)
    else:
        raise ValueError(gt_sfm_model)
    name2id = {image.name: i for i, image in images.items()}

    # 因为拿的是全数据模型做的，所以必须得有测试表
    if test_file_list is None:
        test_names = list(name2id)
    else:
        with open(test_file_list, 'r') as f:
            test_names = f.read().rstrip().split('\n')

    # translation and rotation errors
    analysis_list = {}

    z = np.array([[0, 0, 0, 1]])

    # diff车库 db->dif转换矩阵
    # trans = np.array([[0.306034, 0.0379599, -0.92952, 0.0139929],
    #                 [-0.03399396, 0.978978, 0.0279279, 0.0009009],
    #                 [0.929929, 0.02299297, 0.3079076, -0.0180009],
    #                 [0, 0, 0, 1]])
    # 车库 db->full转换矩阵
    # trans = np.array([[0.915, 0.012, -0.373, -0.046],
    #                   [-0.024,  0.987, -0.029,  0.],
    #                   [0.372, 0.036,  0.915, 0.026],
    #                   [0.,    0.,     0.,   1.]])
    # 905办公室 db->full转换矩阵
    # trans = np.array([[0.939748, -0.034691, 0.250936, 0.211898],
    #                   [0.009, 0.969, 0.097, 0.005],
    #                   [-0.25294, -0.090966,  0.93575, -0.099213],
    #                   [0.,    0.,     0.,   1.]])
    # 新车库 db->full转换矩阵
    trans = np.array([[1.0230, 0.003, 0.03, -0.039],
                      [-0.003, 1.023, -0.004, 0.001],
                      [-0.03, 0.004,  1.023, -0.033],
                      [0.,    0.,     0.,   1.]])


    errors_t = []
    errors_R = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.
        else:
            image = images[name2id[name]]
            R_gt, t_gt = image.qvec2rotmat(), image.tvec
            qvec, t = predictions[name]
            R = qvec2rotmat(qvec)
            t = -R.T @ t
            R = R.T
            tem1 = np.c_[R, t]
            tem2 = np.r_[tem1, z]
            final = np.matmul(trans, tem2)
            t = final[:-1, -1]
            R = final[:-1, :-1]

            t_gt = -R_gt.T @ t_gt
            R_gt = R_gt.T
            e_t = np.linalg.norm(-t_gt + t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))

        # lys_table_241数据 测得箱子为68cm,模型中为1.09m，比例尺为0.68/1.05=0.623853211
        # lys_241_new数据 测得箱子为32cm，模型中为42，比例尺为31/42=0.761904
        # lys_905数据 测得走廊为162cm,模型中为0.78m，比例尺为162/78=2.076
        # lys_地下车库数据，测得门为150cm，角落为300cm,模型中为30cm,60cm，比例尺为5
        # lys_鱼眼905数据，测得俩座位宽为197cm,模型中为0.88m，比例尺为2.238
        # jiawei_905数据，测得座位之间为202cm，模型中为59cm，比例尺为3.4237
        # jiawei_cheku数据，比例尺为4.828
        # jiawei_cheku_dif数据，比例尺为5.02
        # 905 比例尺为2.15
        # 新车库4.638
        e_t *= 4.638
        errors_t.append(e_t)
        errors_R.append(e_R)
        analysis_list[name] = {"error_R": e_R, "error_t": e_t}

    _1, _2 = cumulative_recall(errors_t)
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    med_t = np.median(errors_t)
    med_R = np.median(errors_R)

    recall = {'1': _2[[idx-1 for idx, x in enumerate(_1) if(x > 1)][0]],
              '0.5': _2[[idx-1 for idx, x in enumerate(_1) if(x > 0.5)][0]],
              '0.2': _2[[idx-1 for idx, x in enumerate(_1) if(x > 0.2)][0]],
              '0.1': _2[[idx-1 for idx, x in enumerate(_1) if(x > 0.1)][0]],
              '0.05': _2[[idx-1 for idx, x in enumerate(_1) if(x > 0.05)][0]]
              }

    lys_mean = [x for x in errors_t if x < 30]
    mean_t = np.mean(lys_mean)
    mean_R = np.mean(errors_R)
    out = f'\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg'

    out += '\nPercentage of test images localized within:'
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f'\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%'
    logger.info(out)
    print("----------------------------------------------------------")

    threshs_t = [0, 0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    out = "\n\t\t\ttotal\t\t0deg--1deg\t1edg--2edg\t2edg--3edg\t3edg--5edg\t5deg--7deg\t7deg--10deg\t大于10deg"
    for t in range(len(threshs_t)):
        if(t == 0):
            continue

        # 每排的开头，总和部分
        out += f"\n\t{threshs_t[t-1]*100:.0f}cm--{threshs_t[t]*100:.0f}cm:"
        temp = (errors_t < threshs_t[t]) & (errors_t > threshs_t[t-1])
        ratio = np.mean(temp)
        out += f"\t{ratio*100:.2f}%\t"

        # 细分到每个度数的区间
        for r in range(len(threshs_R)):
            if(r == 0):
                continue
            ratio = np.mean(temp & (errors_R < threshs_R[r]) & (
                errors_R > threshs_R[r-1]))
            out += f"\t{ratio*100:.2f}%\t"
            if(r == (len(threshs_R)-1)):
                ratio = np.mean(temp & (errors_R > threshs_R[r]))
                out += f"\t{ratio*100:.2f}%"

        # 最后一排的另算
        if(t == (len(threshs_t)-1)):
            temp = (errors_t > threshs_t[t])
            ratio = np.mean(temp)
            out += f"\n\t大于{threshs_t[t]*100:.0f}cm:"
            out += f"\t{ratio*100:.2f}%\t"
            for r in range(len(threshs_R)):
                if(r == 0):
                    continue
                ratio = np.mean(temp & (errors_R < threshs_R[r]) & (
                    errors_R > threshs_R[r-1]))
                out += f"\t{ratio*100:.2f}%\t"
                if(r == (len(threshs_R)-1)):
                    ratio = np.mean(temp & (errors_R > threshs_R[r]))
                    out += f"\t{ratio*100:.2f}%"
    out += "\n\t\t\t\t"
    for r in range(len(threshs_R)):
        if(r == 0):
            continue
        temp = (errors_R < threshs_R[r]) & (errors_R > threshs_R[r-1])
        ratio = np.mean(temp)
        out += f"\t{ratio*100:.2f}%\t"
    temp = (errors_R > threshs_R[-1])
    ratio = np.mean(temp)
    out += f"\t{ratio*100:.2f}%\t"
    print(out)


def cumulative_recall(errors: np.ndarray) -> Tuple[np.ndarray]:
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    return errors, recall*100
