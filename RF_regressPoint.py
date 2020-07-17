from utils import *
import numpy as np
import json
import os
import cv2
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from math import *




root_dir = 'D:/Data_set/speed'
partitions, labels = process_json_dataset(root_dir)
'''返回两个如下字典，partitions存文件名
partitions = {'test': [], 'train': [], 'real_test': []}
labels = {文件名：{‘q’:[],'r':[]},文件名2：{‘q’:[],'r':[]}}
'''
img_label = []  # 储存label的列表,特征和label一一对应

# 将数据分为特征集和标签集
text = file_name(root_dir)
img_feature = np.zeros((1, 1410))
index = []
for i in range(1000):
    path = text[i]  # 按顺序提取图片路径
    out1 = feature_g(path)
    img_name = partitions['train'][i]  # 获取训练集第i张图片的文件名
    img_index = int(img_name.split('.')[0].split('img')[1])  # 获取文件名中的数字

    if out1 is not None:    # 判断传回的特征是否为None，是则说明此图片放弃，否则加入特征向量中
        img_feature = np.vstack((img_feature, out1))
        label = np.hstack((img_index, labels[img_name]['q'] + labels[img_name]['r']))
        img_label.append(label)  # q,r拼接成一个列表
    else:
        index = np.hstack((index, i))

img_feature = np.delete(img_feature, 0, 0)

img_feature = np.array(img_feature)  # 把列表转换为array（下边函数形参输入为数组）
img_label = np.array(img_label)

# 拆分数据集为训练集 测试集
X_train, X_test, y_train_pr, y_test_qr = train_test_split(img_feature, img_label, test_size=0.15, random_state=4)
#
print('train size:', X_train.shape, 'Test size:', X_test.shape)

# 对11个点进行回归

projectData = []# 用一个列表储存验证集PnP求解出的q和r
projectTrainData = []# 用一个列表储存训练集PnP求解出的q和r
points3D = np.array([[0.5446,0.4894,0.2551],
                     [0.3025,-0.5799,0.2526],
                     [-0.5450,0.4882,0.2599],
                     [0.3697,0.2996,0.0015],
                     [0.3687,-0.2585,0.0019],
                     [-0.3626,-0.2590,0.0014],
                     [-0.3635,0.3019,0.0016],
                     [0.3671,0.3831,0.3187],
                     [0.3668,-0.3824,0.3172],
                     [-0.3720,-0.3824,0.3162],
                     [-0.3696,0.3822,0.3196]])# 读取11个三维点坐标
for point in points3D:
    # 对于每个点都重新生成一组label，用于关键点回归。每个label由图像名字和当前三维点在图像上的投影[file_name, u, v]构成
    y_train = []
    y_test = []

    for train_qr in y_train_pr:
        file = train_qr[0]
        q = train_qr[1:5]
        r = train_qr[5:]
        u, v = project4any(point, q, r)
        y_train.append([file, u, v])

    for test_qr in y_test_qr:
        file = test_qr[0]
        q = test_qr[1:5]
        r = test_qr[5:]
        u, v = project4any(point, q, r)

        y_test.append([file, u, v])

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 随机森林回归
    regr_rf = RandomForestRegressor(n_estimators=20, max_depth=20, random_state=2)  # 随机森林回归模型
    regr_rf.fit(X_train, y_train[:, 1:])  # 训练模型
    # 预测测试集上样本
    y_rf = regr_rf.predict(X_test)#验证集结果
    y_tr = regr_rf.predict(X_train)#训练集结果
    #储存结果用于PnP
    projectData.append(y_rf)
    projectTrainData.append(y_rf)


projectData = np.array(projectData)# 存储结构[[第1个点在所有图像上回归结果]，[第2个点在所有图像上回归结果]] 11 * test_num * 2
projectTrainData = np.array(projectTrainData)

# 用PnP求解q和r
projectData = projectData.transpose([1,0,2])# 将projectData形状转换为test_num * 11 * 2
projectTrainData = projectTrainData.transpose([1,0,2])

K = Camera.K
dist_coeffs = np.zeros((4, 1))
qr_pre = []# 存储所有验证集的q和r

# 验证集预测的q和r
for imgPoint in projectData:
    #print('check input shape: ', imgPoint.shape, points3D.shape)
    imgPoint = np.ascontiguousarray(imgPoint)# Opencv solvepnp要求objectPoint,imgPoint都在一块连续内存上
    _, rvec, tvec = cv2.solvePnP(points3D, imgPoint, K, distCoeffs=0, flags=cv2.SOLVEPNP_ITERATIVE)
    rvec = rvec.transpose().squeeze()# 将列向量转换为行向量的形式
    tvec = tvec.transpose().squeeze()

    R = Rotation.from_rotvec(rvec)
    q = R.as_quat()[[3,1,2,0]]# 将四元数转换为[w, x, y, z]的形式

    qr_pre.append(np.hstack((q,tvec)))

qr_pre = np.array(qr_pre)

qr_TrainPre = []# 存储所有训练集的q和r

# 训练集预测的q和r
for imgPoint in projectTrainData:
    #print('check input shape: ', imgPoint.shape, points3D.shape)
    imgPoint = np.ascontiguousarray(imgPoint)# Opencv solvepnp要求objectPoint,imgPoint都在一块连续内存上
    _, rvec, tvec = cv2.solvePnP(points3D, imgPoint, K, distCoeffs=0, flags=cv2.SOLVEPNP_ITERATIVE)
    rvec = rvec.transpose().squeeze()# 将列向量转换为行向量的形式
    tvec = tvec.transpose().squeeze()

    R = Rotation.from_rotvec(rvec)
    q = R.as_quat()[[3,1,2,0]]# 将四元数转换为[w, x, y, z]的形式

    qr_TrainPre.append(np.hstack((q,tvec)))

qr_TrainPre = np.array(qr_TrainPre)

# 计算测试集score
score_list = []
for qr_t, qr_p in zip(y_test_qr, qr_pre):
    qr_gt = qr_t[1:]
    r_gt = qr_gt[4:]  # 标签中后三个元素是位置
    r_pre = qr_p[4:]

    score_posi = sqrt(np.sum((r_gt - r_pre) ** 2)) / sqrt(np.sum(r_gt ** 2))

    q_gt = qr_gt[:4]  # 标签中前四个元素是角度
    q_pre = qr_p[:4]

    score_orient = 2 * acos(abs(np.sum(q_gt * q_pre)))
    score = score_posi + score_orient

    score_list.append([score] + list(qr_t) + list(qr_p))# score的储存结构式[score, file_name, qr_t, qr_p]

score_result = sorted(score_list, key=lambda x:x[0])# 按照score排序
score_result = np.array(score_result)
print('test score: ', np.mean(score_result[:, 0]))# 输出平均score

# 计算训练集score
score_list = []
for qr_t, qr_p in zip(y_train_pr, qr_TrainPre):
    qr_gt = qr_t[1:]
    r_gt = qr_gt[4:]  # 标签中后三个元素是位置
    r_pre = qr_p[4:]

    score_posi = sqrt(np.sum((r_gt - r_pre) ** 2)) / sqrt(np.sum(r_gt ** 2))

    q_gt = qr_gt[:4]  # 标签中前四个元素是角度
    q_pre = qr_p[:4]

    score_orient = 2 * acos(abs(np.sum(q_gt * q_pre)))
    score = score_posi + score_orient

    score_list.append([score] + list(qr_t) + list(qr_p))# score的储存结构式[score, file_name, qr_t, qr_p]

score_result = sorted(score_list, key=lambda x:x[0])# 按照score排序
score_result = np.array(score_result)
print('train score: ', np.mean(score_result[:, 0]))# 输出平均score
