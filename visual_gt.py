#！/usr/bin/env python
# -*- coding: utf-8 -*-
# @ Author  : Jessica
# @ Time    : 2020/12/12 16:32
# @File     : visual_gt.py
# @Software : PyCharm
# @file function: GT映射到原图轮廓
import cv2
import numpy as np
import os

color=(200,0,0)
mask_path='/data/jessica/work2/egc2/mask/'
img_path='D:\YZU-capstone\ISIC2018_Task1-2_Test_Input\ISIC_0012236.jpg'
visual_path='D:\YZU-capstone\CA-Net\VIS'
if not os.path.isdir(visual_path):
    os.makedirs(visual_path)

def visual_gt(img,mask,color,path_visual):
    h,w=mask.shape
    mask_3d=np.ones((h,w),dtype='uint8')*255
    mask_3d[mask[:,:]==0]=0
    ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 4)
    cv2.imwrite(path_visual, img)

    return img

def visusal_all(mask_path,img_path,visual_path,color):
    name_list=os.listdir(img_path)
    for i in range(len(name_list)):
        img=cv2.imread(os.path.join(img_path,name_list[i]))
        mask=cv2.imread(os.path.join(mask_path,name_list[i]),cv2.IMREAD_GRAYSCALE)
        path_visual=os.path.join(visual_path,name_list[i])
        visual_gt(img,mask,color,path_visual)



visusal_all(mask_path,img_path,visual_path,color)

