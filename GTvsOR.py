#！/usr/bin/env python
# -*- coding: utf-8 -*-
# @ Author  : Jessica
# @ Time    : 2020/10/19 19:07
# @File     : GTvsOR.py
# @Software : PyCharm
# @file function: GT在原图显示出轮廓
import cv2
import os
import numpy as np
visual_path='/home/dwj/pytorch_xuexi/expt/yy/'
imgfile = '/home/dwj/pytorch_xuexi/expt/gtimg/335_1325.jpg'
maskfile = '/home/dwj/pytorch_xuexi/expt/gt/335_1325.jpg'
if not os.path.isdir(visual_path):
    os.makedirs(visual_path)

import matplotlib.pyplot as plt
image=cv2.imread(imgfile)
mask_2d=cv2.imread(maskfile,cv2.IMREAD_GRAYSCALE)
# cv2.imshow("zd",mask_2d)
h,w=mask_2d.shape
# print(h,w)
mask_3d=np.ones((h,w),dtype='uint8')*255#全255的np
# print('nn',mask_3d.shape)
mask_3d[mask_2d[:,:]==0]=0
ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]
cv2.drawContours(image, contours, -1, (192, 0, 0), 4)
# 打开画了轮廓之后的图像
print('imhsize',image.shape)
# cv2.imshow('mask', image)

# k = cv2.waitKey(0)
# if k == 27:
#     cv2.destroyAllWindows()
# cv2.imshow('mask', image)
# 保存图像

cv2.imwrite(os.path.join(visual_path,"showedges1.png"), image)
# def union_image_mask(image_path, mask_path, num):
#     # 读取原图
#     image = cv2.imread(image_path)
#     # print(image.shape) # (400, 500, 3)
#     # print(image.size) # 600000
#     # print(image.dtype) # uint8
#
#     # 读取分割mask，这里本数据集中是白色背景黑色mask
#     mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     # 裁剪到和原图一样大小
#     mask_2d = mask_2d[0:400, 0:500]
#     h, w = mask_2d.shape
#     cv2.imshow("2d", mask_2d)
#
#     # 在OpenCV中，查找轮廓是从黑色背景中查找白色对象，所以要转成黑色背景白色mask
#     mask_3d = np.ones((h, w), dtype='uint8')*255
#     # mask_3d_color = np.zeros((h,w,3),dtype='uint8')
#     mask_3d[mask_2d[:, :] == 255] = 0
#     cv2.imshow("3d", mask_3d)
#     ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
#     im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cnt = contours[0]
#     cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1)#可改颜色
#     # 打开画了轮廓之后的图像
#     cv2.imshow('mask', image)
#     k = cv2.waitKey(0)
#     if k == 27:
#         cv2.destroyAllWindows()
#     # 保存图像
#     # cv2.imwrite("./image/result/" + str(num) + ".bmp", image)
