from asyncio.proactor_events import _ProactorBaseWritePipeTransport
from pyexpat import model
from pixloc.utils.colmap import *
from pixloc.utils.io import parse_image_list
import torch
import torchvision
import torch.nn as nn
from pixloc.utils.quaternions import qvec2rotmat, rotmat2qvec
# -----------------------------------------------------------
# 归一化point3D文件
# path = '/home/lys/Workplace/datasets/jiawei_cheku_outputs/colmap_dif/sfm_superpoint+superglue'
# cameras, dbs, points3D = read_model(path)

# with open(path+"/point3D_normal.txt", 'w') as f:
#     for i in points3D:
#         xyz, rgb = points3D[i].xyz, points3D[i].rgb
#         f.write(str(xyz[0])+' '+str(xyz[1])+' '+str(xyz[2]) +
#                 ' '+str(rgb[0])+' '+str(rgb[1])+' '+str(rgb[2])+"\n")
# print(1)
# -----------------------------------------------------------
# 用于测试torch.clamp函数
# a= (torch.randn((3,4))+1)*10
# a
# print()
# -----------------------------------------------------------
# 加载预训练模型和权重
# model1=torchvision.models.vgg19(pretrained=True)
# i=model1.features[0]
# print(i)
# print("weight  ",i.weight)
# print("bias  ",i.bias)
# # level-1
# feature0 =model1.features[0]
# bias0=model1.features[1]
# feature1=model1.features[2]
# print()
# -----------------------------------------------------------
# 将colmap_db中的轨迹->trans->colmap中
# datasetsname = 'jiawei_cheku'

# front = '/home/lys/Workplace/datasets/'
# back = '_outputs/colmap_db/sfm_superpoint+superglue'
# path = front+datasetsname+back
# cameras, dbs, points3D = read_model(path)

# for i in dbs:
#     print()
# -----------------------------------------------------------
# 测试从colmap_db到colmap_full的变换
# a=np.array([[1,2,3],[3,1,2],[4,1,3]])
# qvec=rotmat2qvec(a)
# R=qvec2rotmat(qvec)
# qvec1=rotmat2qvec(R)
# if(qvec.all()==qvec1.all()):
#     print("true")
# b=np.array([[1,3],[0,1]])
# print(np.matmul(a,b))

# path = '/home/lys/Workplace/datasets/jiawei_cheku_outputs/colmap_full/sfm_superpoint+superglue'
# # path="/home/lys/Workplace/datasets/jiawei_cheku_outputs/results/pixloc_outputs.txt"
# cameras, dbs, points3D = read_model(path)
# db_qvec, db_t = dbs[125].qvec, dbs[125].tvec
# print(dbs[125].name)
# db_R = qvec2rotmat(db_qvec)
# db_t_real = -db_R.T @ db_t
# db_R=db_R.T
# # db_R[...,1:]*=-1
# qvec1=rotmat2qvec(db_R)
# test_q=[qvec1[1],qvec1[2],qvec1[3],qvec1[0]]
# test_t=db_t_real*4.828
# rot1=qvec2rotmat(qvec1)
# qvec2=rotmat2qvec(rot1)
# rot2=qvec2rotmat(qvec2)
# print()
# z = np.array([[0, 0, 0, 1]])
trans_1 = np.array([[1.023, 0.003, 0.030, -0.036],
                    [-0.003, 1.023, -0.004, 0.001],
                    [-0.030, 0.004, 1.023, -0.035],
                    [0, 0, 0, 1]])
trans_2 = np.array([[1, 0, 0, -0.003],
                    [-0, 1, 0, 0],
                    [-0, -0, 1, 0.002],
                    [0, 0, 0, 1]])
# trans_3 = np.array([[0.913,0.012,-0.369,-0.046
# -0.025 0.984 -0.029 -0.001
# 0.368 0.036 0.912 0.02
# 0 0 0 1])
# final1=np.dot(trans_2,trans_1)
# print(final1)
final = np.matmul(trans_2, trans_1)
print(final)
print("")
# trans=np.c_[db_R, db_t_real]
# f=np.r_[trans, z]
# fi=np.matmul(trans_no,f)
# yuanlai_t = fi[:-1,-1]
# yuanlai_R = fi[:-1, :-1]
# datasetsname = 'jiawei_cheku'

# front = '/home/lys/Workplace/datasets/'
# back = '_outputs/colmap_full/sfm_superpoint+superglue'
# path = front+datasetsname+back
# cameras, dbs, points3D = read_model(path)
# db_qvec, db_t = dbs[1].qvec, dbs[1].tvec
# full_R = qvec2rotmat(db_qvec)

# full_t_real = -full_R.T @ db_t
# full_R=full_R.T
# e_t = np.linalg.norm(-full_t_real + yuanlai_t, axis=0)*4.828
# cos = np.clip((np.trace(np.dot(full_R.T,yuanlai_R)) - 1) / 2, -1., 1.)
# e_R = np.rad2deg(np.abs(np.arccos(cos)))


# datasetsname = 'jiawei_cheku'

# front = '/home/lys/Workplace/datasets/'
# back = '_outputs/colmap_full/sfm_superpoint+superglue'
# path = front+datasetsname+back
path = '/home/lys/Workplace/datasets/jiawei_cheku_outputs/colmap_full/sfm_superpoint+superglue'
# # path="/home/lys/Workplace/datasets/jiawei_cheku_outputs/results/pixloc_outputs.txt"
cameras, dbs, points3D = read_model(path)

with open("full_camera_track.txt", 'w') as f:
    for i in dbs:
        db_qvec, db_t = dbs[i].qvec, dbs[i].tvec
        db_R = qvec2rotmat(db_qvec)
        db_t_real = -db_R.T @ db_t
#         db_R=db_R.T
#         db_R[...,1:]*=-1
#         qvec=rotmat2qvec(db_R)
# #         # trans=np.c_[db_R, db_t_real]
# #         trans = np.r_[db_t_real, 1]
# #         full = np.matmul(trans_no,trans)
# #         t_full = full[:-1,-1]
# #         q_full = full[:-1, :-1]
# #         # t_real = -q_full.T @ t_full
# #         # t_real=t_full
# #     # print(dbs[i].name+' '+str(dbs[i].qvec[0])+' '+str(dbs[i].qvec[1])+' '+str(dbs[i].qvec[2])+' '+str(dbs[i].qvec[3])+' '+str(t_real[0])+' '+str(t_real[1])+' '+str(t_real[2])+'\n')
        f.write(dbs[i].name+' '+str(db_t_real[0])+' ' +
                str(db_t_real[1])+' '+str(db_t_real[2])+"\n")
print(1)
# with open("new_query.txt",'w') as n:
#     predictions = parse_image_list("pixloc_outputs.txt", with_poses=True)
#     predictions = {n: (im.qvec, im.tvec) for n, im in predictions}
#     for name in predictions:
#         qvec, t = predictions[name]
#         R = qvec2rotmat(qvec)
#         t=-R.T @ t
#         n.write(name+' '+str(qvec[0])+' '+str(qvec[1])+' '+str(qvec[2])+' '+str(qvec[3])+' '+str(t[0])+' '+str(t[1])+' '+str(t[2])+"\n")

# -----------------------------------------------------------
# datasetsname='jiawei_cheku'
# front='/home/lys/Workplace/datasets/'
# back='_outputs/colmap_full/sfm_superpoint+superglue'
# path='/home/lys/Workplace/datasets/test'
# cameras, dbs, points3D = read_model(path)
# name2id = {image.name: i for i, image in dbs.items()}
# write_model(cameras,dbs,points3D,path,ext=".txt")
# with open("db_gt_qt.txt",'r') as n:
# predictions = parse_image_list("db_gt_qt.txt", with_poses=True)
# predictions = {n: (im.qvec, im.tvec) for n, im in predictions}
# for name in predictions:
#     qvec, t = predictions[name]
#     R = qvec2rotmat(qvec)
#     if(name=='db/20220911_135748.JPG'):
#         print(1)
#     if(name=='db/20220911_140314.JPG'):
#         print(1)
#     if(name=='db/20220911_141457.JPG'):
#         print(1)
#     if(name=='db/20220911_143537.JPG'):
# print(1)
# t=-R.T @ t
# t=np.r_[t, 1]
# t=np.matmul(trans_no,t)
# n.write(name+' '+str(t[0])+' '+str(t[1])+' '+str(t[2])+"\n")

# with open("db_gt_qt.txt",'r') as f:
#     for i in dbs:
#         # print(dbs[i].name+' '+str(dbs[i].qvec[0])+' '+str(dbs[i].qvec[1])+' '+str(dbs[i].qvec[2])+' '+str(dbs[i].qvec[3])+str(dbs[i].tvec[0])+' '+str(dbs[i].tvec[1])+' '+str(dbs[i].tvec[2]))
#         qvec, t = dbs[i].qvec,dbs[i].tvec
#         R = qvec2rotmat(qvec)
#         t=-R.T @ t
#         f.write(dbs[i].name+' '+str(t[0])+' '+str(t[1])+' '+str(t[2])+"\n")

# -----------------------------------------------------------
# colmap输出bin->txt
# write_model(cameras,dbs,points3D,path,ext=".txt")
# print('sfm模型中共含有')
# print(name2id.__len__())
# print("写入成功")
# -----------------------------------------------------------
