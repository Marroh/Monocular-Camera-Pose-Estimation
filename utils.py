import numpy as np
import json
import os
import cv2
import random
import matplotlib.pyplot as plt

class Camera:
    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx, 0, nu / 2],
         [0, fpy, nv / 2],
         [0, 0, 1]]
    K = np.array(k)

def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'test.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)

    partitions = {'test': [], 'train': [], 'real_test': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])

    return partitions, labels

# 提取目标文件夹下的所有jpg图片文件的路径
def file_name(file_dir):
    path1 = []
    for root, dirs, files in os.walk(file_dir):
        if 'train' in dirs:
            for file in os.listdir(os.path.join(root, 'train')):
                if os.path.splitext(file)[1] == '.jpg':
                    path1.append(os.path.join(root, 'train', file))
    return path1


# 在进行了大津阈值二值化后，对图片寻找卫星区域，得到区域的四个角的row和col，求取面积和长宽比
def area_ax(th):
    row = th.shape[0]
    col = th.shape[1]
    is_first = 0
    begin_row = 0
    end_row = 0
    begin_col = 0
    end_col = 0
    for i in range(row):
        for j in range(col):
            if is_first == 0:
                if th[i][j] == 255:
                    begin_row = i
                    end_row = i
                    begin_col = j
                    end_col = j
                    is_first = 1
            else:
                if th[i][j] == 255:
                    if i < begin_row:
                        begin_row = i
                    if i > end_row:
                        end_row = i
                    if j < begin_col:
                        begin_col = j
                    if j > end_col:
                        end_col = j
    return begin_row, end_row, begin_col, end_col


# 计算欧几里得距离
def distEclud(vec_a, vec_b):
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


def randCent(dataset, k):
    n = np.shape(dataset)[1]
    n1 = np.shape(dataset)[0]
    inter = np.floor(n1 / k)
    centroids = np.mat(np.zeros((k, n)))
    low1 = 0
    high1 = inter - 1
    for j in range(k):
        ran = np.random.randint(low1, high1, 1)
        low1 = low1 + inter
        high1 = high1 + inter
        centroids[j, :] = dataset[ran, :]
    return centroids


def k_means(dataset, k, distMeans = distEclud, createCent = randCent):
    m = np.shape(dataset)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataset, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for p in range(m):
            mindist = 1000000000
            minindex = -1
            for q in range(k):
                dist_q = distMeans(centroids[q, :], dataset[p, :])
                if dist_q < mindist:
                    mindist = dist_q
                    minindex = q
            if clusterAssment[p, 0] != minindex:
                clusterChanged = True
                clusterAssment[p, :] = minindex, mindist ** 2
        for cent in range(k):
            store = dataset[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            if len(store) != 0:
                centroids[cent, :] = np.mean(store, axis=0)
    label = clusterAssment[:, 0]
    return label, centroids


def axis_change(a, begin_row, end_row):
    n = len(a)
    b = np.zeros((n, 2))
    for i in range(n):
        b[i, 1] = end_row - begin_row - a[i, 1]
        b[i, 0] = a[i, 0]
    return b


def sort_num(store, p):
    k = len(store)
    num = 0
    for i in range(k):
        if store[i] < store[p]:
            num = num + 1
    return num


def sort_center(center, s, begin_col, end_col):
    store = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(11):
        store[i] = center[i, 0] + center[i, 1] * (end_col - begin_col)
    s1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(11):
        num = sort_num(store, i)
        s1[num] = s[i]
    return s1


# 提取特征，sift点提取，利用k-mean对关键点坐标进行分类，得到11个类别，然后再在11个类中，随机取样，得到11个随机取样点，特征为这11个取样点的描述子
def feature_g(path_):
    sift = cv2.xfeatures2d.SIFT_create()

    img1 = cv2.imread(path_)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(img1, 0, 15, 5)

    ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    begin_row, end_row, begin_col, end_col = area_ax(th2)
    image = np.mat(image)[begin_row:end_row, begin_col:end_col]
    # image = cv2.bilateralFilter(image, 0, 15, 5)

    # 卫星面积特征
    area = (end_row - begin_row) * (end_col - begin_col)
    # 卫星长宽比
    lengh_width = (end_col - begin_col) / (end_row - begin_row)

    kp1, des1 = sift.detectAndCompute(image, None)   # kp1是关键点坐标，des是描述子

    # img3 = cv2.drawKeypoints(image, kp1, image, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    # cv2.imshow("s", img3)
    # cv2.waitKey(1)

    points2f = cv2.KeyPoint_convert(kp1)  # 将KeyPoint格式数据中的xy坐标提取出来。

    z = np.float32(points2f)
    z = np.mat(z)
    # k-mean分类，随机取样，判断如果sift点小于11的两倍个分类数，则此训练图片对比度过低，放弃此训练图片
    if len(z) > 11*2:
        # define criteria and apply kmeans()
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # ret, label, center = cv2.kmeans(z, 11, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        label, center = k_means(z, 11, distMeans=distEclud, createCent=randCent)
        label = np.array(label)
        center = np.array(center)

        a0 = z[label.ravel() == 0]
        a1 = z[label.ravel() == 1]
        a2 = z[label.ravel() == 2]
        a3 = z[label.ravel() == 3]
        a4 = z[label.ravel() == 4]
        a5 = z[label.ravel() == 5]
        a6 = z[label.ravel() == 6]
        a7 = z[label.ravel() == 7]
        a8 = z[label.ravel() == 8]
        a9 = z[label.ravel() == 9]
        a10 = z[label.ravel() == 10]

        n = len(z)
        out = []
        s1 = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]

        s = sort_center(center, s1, begin_col, end_col)

        for k in range(11):
            now_ar = s[k]
            num = len(now_ar)
            out_ = np.zeros(128)
            for i in range(int(np.ceil(num / 5))):
                num1 = random.randint(0, num - 1)
                point = now_ar[num1]
                for j in range(n):
                    if point[0, 0] == z[j, 0] and point[0, 1] == z[j, 1]:
                        out_ = out_ + des1[j]
                        break
            out_ = out_ / int(np.ceil(num / 5))
            out = np.hstack((out, out_))
        out = np.hstack((out, area, lengh_width))
    else:
        out = None
    return out


def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q / np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def project(q, r):
    """ Projecting points to image frame to draw axes """

    # reference points in satellite frame for drawing axes
    p_axes = np.array([[0, 0, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 0, 1],
                       [0, 0, 1, 1]])
    points_body = np.transpose(p_axes)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)# 原点，x轴，y轴，z轴

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    # projection to image plane
    points_image_plane = Camera.K.dot(points_camera_frame)

    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y

def project4any(vector, q, r):
    vector = np.array(vector)# vector = [x, y ,z]
    vector = np.hstack((vector, 1))

    # 计算外参矩阵
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, vector.T)# 原点，x轴，y轴，z轴

    # 齐次坐标归一化
    points_camera_frame = p_cam / p_cam[2]

    # 投影到图像平面
    points_image_plane = Camera.K.dot(points_camera_frame)

    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y


def visualize(root, y_test, y_pre, ax_gt=None, ax_pre=None):
    """ Visualizing image, with ground truth pose with axes projected to training image. """

    if ax_gt is None :
        ax_gt = plt.gca()
    if ax_pre is None :
        ax_pre = plt.gca()

    img_path = os.path.join(root, 'images/train', 'img' + '%06d' % y_test[0] + '.jpg')
    print(img_path)
    img = plt.imread(img_path)
    ax_gt.imshow(img)
    ax_pre.imshow(img)

    # ground truth可视化
    q, r = y_test[1:5], y_test[5:]
    xa, ya = project(q, r)
    ax_gt.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
    ax_gt.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
    ax_gt.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

    # 预测结果可视化
    q_, r_ = y_pre[:4], y_pre[4:]
    xa_, ya_ = project(q_, r_)
    ax_pre.arrow(xa_[0], ya_[0], xa_[1] - xa_[0], ya_[1] - ya_[0], head_width=30, color='r')
    ax_pre.arrow(xa_[0], ya_[0], xa_[2] - xa_[0], ya_[2] - ya_[0], head_width=30, color='g')
    ax_pre.arrow(xa_[0], ya_[0], xa_[3] - xa_[0], ya_[3] - ya_[0], head_width=30, color='b')

    return