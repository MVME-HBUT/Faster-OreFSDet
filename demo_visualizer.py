# 特征图可视化Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from Visualizer.visualizer import get_local
get_local.activate()
#from detectron2.config import get_cfg
from fewx.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import os

import torch
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

'''
def Have_a_Look(image,x):
    x += 1
    print(x)
    #[N，C，H，w] ->[C，H，w]
    im = np.squeeze(image.detach().cpu().numpy())
    #[C，H，W]->[H，W，C]
    im = np.transpose(im,[1,2,0])
    feature_map_combination=[]
    # channle = np.zeros((30,30))
    # channle = np.zeros((15,15))
    for i in range(1024):
        print(i)
        channle_i = im[: ,:,i]
        
        feature_map_combination.append(channle_i)
        # if i == 95:
        #     channle = channle/96 
    # channle = im[: ,:,0]
    feature_map_sum = sum(one for one in feature_map_combination)/1024
    print(feature_map_sum.shape)
    #查看这一层不同通道的图像，在这里有256层
    plt.figure()
    plt.xticks([]),plt.yticks([])#去除坐标轴plt.axis( 'off' )
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.imshow(feature_map_sum,cmap='gray')
    plt.savefig( '/home/lcheng/FewX-master/1/ '+str(x),dpi=100,bbox_inches='tight', pad_inches = -0.1)
    
    
    img = cv2.imread("/home/lcheng/FewX-master/FewX-master/directory/00019.png") # 原图为31.jpg
    gray = cv2.imread("/home/lcheng/FewX-master/1/ 5.png") # 原图为31.jpg
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #将原图转化为灰度图
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET) # 将灰度图转化为热力图 
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)# 将格式转化为GRB
    heatmap = cv2.resize(heatmap, (300, 300), interpolation=cv2.INTER_AREA)
    # result = cv2.addWeighted(img, 0.6, heatmap, 0.5, 0) # 热力图与原图的叠加
    # cv2.imwrite('/home/lcheng/FewX-master/1/6.png', result) # 将得到的热力图保存为32.jpg
    '''
    
    
def Have_a_Look(image,x):
    x += 1
    print(x)
    #[N，C，H，w] ->[C，H，w]
    im = np.squeeze(image.detach().cpu().numpy())
    #[C，H，W]->[H，W，C]
    im = np.transpose(im,[1,2,0])
    feature_map_combination=[]
    # channle = np.zeros((30,30))
    # channle = np.zeros((15,15))
    for i in range(128):
        print(i)
        channle_i = im[: ,:,i]
        
    #     feature_map_combination.append(channle_i)
    #     # if i == 95:
    #     #     channle = channle/96 
    # # channle = im[: ,:,0]
    #     feature_map_sum = sum(one for one in feature_map_combination)#/128
    #     print(feature_map_sum.shape)
        #查看这一层不同通道的图像，在这里有256层
        plt.figure()
        plt.xticks([]),plt.yticks([])#去除坐标轴plt.axis( 'off' )
        plt.axis('off')
        plt.margins(0,0)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        #plt. subplots_adjust(left=None， bottom=None，right=None，top=None， wspace=None，hspace=None)
        #plt.imshow(feature_map_sum,cmap='gray')
        ratio=1
        img_path = "/home/lcheng/fsod_cen/directory/new/00019.png"
        print("load image from: ", img_path)
        img = Image.open(img_path, mode='r')
        img_h, img_w = img.size[0], img.size[1]
        plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
        # scale the image
        img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
        img = img.resize((img_h, img_w))
        plt.imshow(img, alpha=0.7)
        plt.axis('off')
        # normalize the attention map
        mask = cv2.resize(channle_i, (img_h, img_w))
        normed_mask = mask / mask.max()
        normed_mask = (normed_mask * 255).astype('uint8')
        plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap='jet') # OrRd YlGnBu ,hot_r,Greys
        plt.savefig( '/home/lcheng/fsod_cen/directory/new '+str(i),dpi=100,bbox_inches='tight', pad_inches = -0.1)
    
    
    
    '''# [N, C, H, W] -> [C, H, W]
    im = np.squeeze(image.detach().cpu().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])
    #im_mean = np.mean(im,axis=(2))

    # 查看这一层不同通道的图像，在这里有256层
    plt.figure()
    for i in range(16):
        ax = plt.subplot(6, 6, i+1)
        plt.suptitle(str)
        plt.imshow(im[:, :, i], cmap='gray')
    plt.show()
    plt.savefig( '/home/lcheng/FewX-master/FewX-master '+str(x),dpi=100,bbox_inches='tight', pad_inches = -0.1)
    '''
    
# def Have_a_Look(image,x):
#     x += 1
#     print(x)
#     #[N，C，H，w] ->[C，H，w]
#     im = np.squeeze(image.detach().cpu().numpy())
#     #[C，H，W]->[H，W，C]
#     im = np.transpose(im,[1,2,0])
#     feature_map_combination=[]
#     # channle = np.zeros((30,30))
#     # channle = np.zeros((15,15))
#     for i in range(64):
#         print(i)
#         channle_i = im[: ,:,i]
        
#         feature_map_combination.append(channle_i)
#         # if i == 95:
#         #     channle = channle/96 
#     # channle = im[: ,:,0]
#     feature_map_sum = sum(one for one in feature_map_combination)#/128
#     print(feature_map_sum.shape)
#     #查看这一层不同通道的图像，在这里有256层
#     plt.figure()
#     plt.xticks([]),plt.yticks([])#去除坐标轴plt.axis( 'off' )
#     plt.axis('off')
#     plt.margins(0,0)
#     plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#     #plt. subplots_adjust(left=None， bottom=None，right=None，top=None， wspace=None，hspace=None)
#     #plt.imshow(feature_map_sum,cmap='gray')
#     ratio=1
#     img_path = "/home/lcheng/fsod_cen/directory/00020.png"
#     print("load image from: ", img_path)
#     img = Image.open(img_path, mode='r')
#     img_h, img_w = img.size[0], img.size[1]
#     plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
#     # scale the image
#     img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
#     img = img.resize((img_h, img_w))
#     plt.imshow(img, alpha=0.5)
#     plt.axis('off')
#     # normalize the attention map
#     mask = cv2.resize(feature_map_sum, (img_h, img_w))
#     normed_mask = mask / mask.max()
#     normed_mask = (normed_mask * 255).astype('uint8')
#     plt.imshow(normed_mask, alpha=0.6, interpolation='nearest', cmap='Greys') # OrRd YlGnBu ,hot_r,Greys
#     plt.savefig( '/home/lcheng/fsod_cen/directory/new '+str(x),dpi=100,bbox_inches='tight', pad_inches = -0.1)
    



def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
                             save_original_image=False, quality=200):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    img_path = "/home/lcheng/FewX-master/FewX-master/datasets/coco/train2017/00000.png"
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)
        
        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)

    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=quality)