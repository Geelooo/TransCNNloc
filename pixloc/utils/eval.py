import logging

from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import numpy as np
from pyrsistent import v
from .io import parse_image_list
from .colmap import qvec2rotmat, read_images_binary, read_images_text

logger = logging.getLogger(__name__)


def evaluate(gt_sfm_model: Path, predictions: Union[Dict, Path],
             test_file_list: Optional[Path] = None,
             only_localized: bool = False):
    """Compute the evaluation metrics for 7Scenes and Cambridge Landmarks.
       The other datasets are evaluated on visuallocalization.net
    """
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
    analysis_list={}
    
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
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
           
            

        # lys_table_241数据 测得箱子为68cm,模型中为1.09m，比例尺为0.68/1.05=0.623853211
        # lys_241_new数据 测得箱子为32cm，模型中为42，比例尺为31/42=0.761904
        # lys_905数据 测得走廊为162cm,模型中为0.78m，比例尺为162/78=2.076
        # lys_地下车库数据，测得门为150cm，角落为300cm,模型中为30cm,60cm，比例尺为5
        # lys_鱼眼905数据，测得俩座位宽为197cm,模型中为0.88m，比例尺为2.238
        # jiawei_905数据，测得座位之间为202cm，模型中为59cm，比例尺为3.4237
        # jiawei_cheku数据，比例尺为4.828
        # e_t*=4.828
        errors_t.append(e_t)
        errors_R.append(e_R)
        analysis_list[name]={"error_R":e_R,"error_t":e_t}
    #--------------------------------------------------------------------------------

    with open("/home/lys/Workplace/datasets/office_outputs/pixloc_error.txt", 'w') as f:
        for name in analysis_list:
            unet_v = analysis_list[name]
            unet_v = unet_v['error_t']
            f.write(
                f"{name}\t{round(unet_v,5)}\n")
    #--------------------------------------------------------------------------------

    _1,_2=cumulative_recall(errors_t)
    recall_1=[idx-1 for idx,x in enumerate(_1) if(x>1)]
    recall_05=[idx-1 for idx,x in enumerate(_1) if(x>0.5)]
    recall_02=[idx-1 for idx,x in enumerate(_1) if(x>0.2)]
    recall_01=[idx-1 for idx,x in enumerate(_1) if(x>0.1)]
    recall_005=[idx-1 for idx,x in enumerate(_1) if(x>0.05)]
    recall={'1':100 if len(recall_1)==0 else _2[recall_1[0]],
            '0.5':100 if len(recall_05)==0 else _2[recall_05[0]],
            '0.2':100 if len(recall_02)==0 else _2[recall_02[0]],
            '0.1':100 if len(recall_01)==0 else _2[recall_01[0]],
            '0.05':100 if len(recall_005)==0 else _2[recall_005[0]]
            }
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    lys_mean=[x for x in errors_t if x<2]
    mean_t=np.mean(lys_mean)
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
    threshs_R = [0, 1.0, 2.0, 3.0, 5.0, 7.0 ,10.0]
    out="\n\t\t\ttotal\t\t0deg--1deg\t1edg--2edg\t2edg--3edg\t3edg--5edg\t5deg--7deg\t7deg--10deg\t大于10deg"
    for t in range(len(threshs_t)):
        if(t==0):
            continue

        # 每排的开头，总和部分
        out+=f"\n\t{threshs_t[t-1]*100:.0f}cm--{threshs_t[t]*100:.0f}cm:"
        temp=(errors_t<threshs_t[t]) & (errors_t>threshs_t[t-1])
        ratio=np.mean(temp)
        out+=f"\t{ratio*100:.2f}%\t"

        # 细分到每个度数的区间
        for r in range(len(threshs_R)):
            if(r==0):
                continue
            ratio=np.mean(temp&(errors_R<threshs_R[r])&(errors_R>threshs_R[r-1]))
            out+=f"\t{ratio*100:.2f}%\t"
            if(r==(len(threshs_R)-1)):
                ratio=np.mean(temp&(errors_R>threshs_R[r]))
                out+=f"\t{ratio*100:.2f}%"

        # 最后一排的另算
        if(t==(len(threshs_t)-1)):
            temp=(errors_t>threshs_t[t])
            ratio=np.mean(temp)
            out+=f"\n\t大于{threshs_t[t]*100:.0f}cm:"
            out+=f"\t{ratio*100:.2f}%\t"
            for r in range(len(threshs_R)):
                if(r==0):
                    continue
                ratio=np.mean(temp&(errors_R<threshs_R[r])&(errors_R>threshs_R[r-1]))
                out+=f"\t{ratio*100:.2f}%\t"
                if(r==(len(threshs_R)-1)):
                    ratio=np.mean(temp&(errors_R>threshs_R[r]))
                    out+=f"\t{ratio*100:.2f}%"
    out+="\n\t\t\t\t"
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
