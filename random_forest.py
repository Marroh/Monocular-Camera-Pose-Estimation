import numpy as np
import json
import os
import cv2
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import *
from utils import *

root_dir = 'D:/Data_set/speed'
partitions, labels = process_json_dataset(root_dir)
'''
返回两个如下字典，partitions存文件名
partitions = {'test': [], 'train': [], 'real_test': []}
labels = {文件名：{‘q’:[],'r':[]},文件名2：{‘q’:[],'r':[]}}
'''
img_label = []  # 储存label的列表,特征和label一一对应

# 将数据分为特征集和标签集
text = file_name(root_dir)
img_feature = np.zeros((1, 1410))
index = []
for i in range(1000):
    print('loaded:', i+1)
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
img_label = np.array(img_label)

print('total features: ', img_feature.shape)
print('total label: ', img_label.shape)

img_feature = np.array(img_feature)  # 把列表转换为array（下边函数形参输入为数组）
img_label = np.array(img_label)

# 拆分数据集为训练集 测试集
X_train, X_test, y_train, y_test = train_test_split(img_feature, img_label, test_size=0.15, random_state=4)

print('train size:', X_train.shape, 'Test size:', y_train.shape)

# 随机森林回归
regr_rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=2)  # 随机森林回归模型
regr_rf.fit(X_train, y_train[:, 1:])  # 训练模型
# 预测测试集上样本
y_rf = regr_rf.predict(X_test)

# 计算验证集score
score_list = []
for y_t, y_pre in zip(y_test, y_rf):

    y_gt = y_t[1:]
    r_gt = y_gt[4:]  # 标签中后三个元素是位置
    r_pre = y_pre[4:]

    score_posi = sqrt(np.sum((r_gt - r_pre) ** 2)) / sqrt(np.sum(r_gt ** 2))

    q_gt = y_gt[:4]  # 标签中前四个元素是角度
    q_pre = y_pre[:4]
    unit_q_gt = q_gt / sqrt(q_gt.dot(q_gt.T))  # 模值归一化
    unit_q_pre = q_pre / sqrt(q_pre.dot(q_pre.T))

    score_orient = 2 * acos(abs(np.sum(unit_q_gt * unit_q_pre)))
    score = score_posi + score_orient
    score_list.append([score] + [y_t[0]] + list(y_gt) + list(y_pre))

score_result = sorted(score_list, key=lambda x:x[0])
score_result = np.array(score_result)
print('test score: ', np.mean(score_result[:, 0]))


# 计算验证集score
y_tr = regr_rf.predict(X_train)

score_list = []
for y_t, y_pre in zip(y_train, y_tr):
    y_gt = y_t[1:]
    r_gt = y_gt[4:]  # 标签中后三个元素是位置
    r_pre = y_pre[4:]

    score_posi = sqrt(np.sum((r_gt - r_pre) ** 2)) / sqrt(np.sum(r_gt ** 2))

    q_gt = y_gt[:4]  # 标签中前四个元素是角度
    q_pre = y_pre[:4]
    unit_q_gt = q_gt / sqrt(q_gt.dot(q_gt.T))  # 模值归一化
    unit_q_pre = q_pre / sqrt(q_pre.dot(q_pre.T))
    score_orient = 2 * acos(abs(np.sum(unit_q_gt * unit_q_pre)))
    score = score_posi + score_orient
    score_list.append([score] + [y_t[0]] + list(y_gt) + list(y_pre))

score_result = sorted(score_list, key=lambda x:x[0])
score_result = np.array(score_result)
print('train score: ', np.mean(score_result[:, 0]))

# 可视化
fig, axes = plt.subplots(3, 2, figsize=(8, 8))
for i in range(3):
    y_test = score_result[i, 1:9]
    y_pre = score_result[i, 9:]

    visualize(root_dir, y_test, y_pre, ax_gt=axes[i][0], ax_pre = axes[i][1])
    axes[i][0].axis('off')
    axes[i][1].axis('off')
    fig.tight_layout()

plt.show()
