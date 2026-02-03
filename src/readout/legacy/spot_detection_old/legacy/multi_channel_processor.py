"""
多通道图像处理和spot检测模块
Multi-channel image processing and spot detection module
"""

import os
import glob
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

import cv2
from scipy.spatial import KDTree
from skimage.feature import peak_local_max
import tifffile
from pprint import pprint


def tophat_spots(image, kernel_size=7):
    """应用tophat形态学操作检测spots"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def extract_coordinates(image, local_max_thre=200, intensity_thre=None):
    """从图像中提取坐标点"""
    meta = {}
    coordinates = peak_local_max(image, min_distance=2, threshold_abs=local_max_thre)
    meta['Coordinates brighter than given SNR'] = coordinates.shape[0]
    meta['Image mean intensity'] = float(np.mean(image))
    
    if intensity_thre is not None:
        if intensity_thre <= 1:
            intensities = image[coordinates[:, 0], coordinates[:, 1]]
            meta[f'{intensity_thre} quantile'] = float(np.quantile(intensities, intensity_thre))
            intensity_thre = np.quantile(intensities, intensity_thre)
        coordinates = coordinates[image[coordinates[:, 0], coordinates[:, 1]] > intensity_thre]

    meta['Final spots count'] = coordinates.shape[0]
    return coordinates


def calculate_snr(image, points, neighborhood_size=10, verbose=True):
    """
    计算给定点的信噪比
    SNR定义为点的值除以其邻域的最小值
    """
    offset = -int(-neighborhood_size // 2)
    padded_img = np.pad(image, offset, mode='edge')
    snr_values = []
    
    for point in tqdm(points, disable=not verbose, desc='Calculating snr', position=1, leave=False):
        y, x = point
        x_min, x_max = x, x + neighborhood_size
        y_min, y_max = y, y + neighborhood_size
        
        neighborhood = padded_img[y_min:y_max, x_min:x_max]
        min_val = np.min(neighborhood)
        point_val = image[y, x]
        snr = point_val / min_val if min_val != 0 else float('inf')
        snr_values.append(snr)
    
    return snr_values


def extract_signal(image_big, pad_x, cut_x, pad_y, cut_y, 
                   tophat_mean, snr=8, abs_thre=200,
                   tophat_break=100, intensity_thre=None,
                   check_snr=False, kernel_size=5):
    """从图像块中提取信号"""
    image_raw = image_big[pad_y: pad_y + cut_y, pad_x: pad_x + cut_x]
    
    # tophat spots
    image = tophat_spots(image_raw, kernel_size)
    image[image < tophat_break] = 0

    # extract coordinates
    if abs_thre is None:
        coordinates = extract_coordinates(image, local_max_thre=tophat_mean * snr, intensity_thre=intensity_thre)
    else:
        coordinates = extract_coordinates(image, local_max_thre=min(abs_thre, tophat_mean * snr), intensity_thre=intensity_thre)

    if check_snr:
        snr_values = calculate_snr(image_raw, coordinates)
    else:
        snr_values = None
    del image_raw

    # find signal
    Maxima = np.zeros(image.shape, dtype=np.uint16)
    Maxima[coordinates[:, 0], coordinates[:, 1]] = 255
    image[Maxima <= 0] = 0  # Mask

    # dilation of image
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return coordinates, snr_values, image


def read_intensity(image_dict, coordinates, channel, snr=None):
    """读取坐标点的强度值"""
    if snr is None:
        intensity = pd.DataFrame({
            'Y': coordinates[:, 0], 
            'X': coordinates[:, 1], 
            'Channel': [channel] * len(coordinates)
        })
    else:
        intensity = pd.DataFrame({
            'Y': coordinates[:, 0], 
            'X': coordinates[:, 1], 
            'Channel': [channel] * len(coordinates), 
            'snr': snr
        })

    for image_name, image in image_dict.items():
        intensity[image_name] = image[coordinates[:, 0], coordinates[:, 1]]
    return intensity


def remove_duplicates(coordinates):
    """移除重复的坐标点"""
    tree = KDTree(coordinates)
    pairs = tree.query_pairs(2)
    
    neighbors = {}
    for i, j in pairs:
        if i not in neighbors:
            neighbors[i] = set([j])
        else:
            neighbors[i].add(j)
        if j not in neighbors:
            neighbors[j] = set([i])
        else:
            neighbors[j].add(i)

    keep = []
    discard = set()
    nodes = set([s[0] for s in pairs] + [s[1] for s in pairs])
    
    for node in nodes:
        if node not in discard:
            keep.append(node)
            discard.update(neighbors.get(node, set()))
    
    centroids_simplified = np.delete(coordinates, list(discard), axis=0)
    return centroids_simplified


def divide_main(shape, max_volume=10**8, overlap=500, data_dict=None, verbose=True):
    """将大图像分割成小块进行处理的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if len(shape) == 3:
                zrange, xrange, yrange = shape
            elif len(shape) == 2:
                yrange, xrange = shape
                zrange = 1

            xy_size = int(np.sqrt(max_volume / zrange))
            x_num = -(-(xrange - overlap) // (xy_size - overlap))
            y_num = -(-(yrange - overlap) // (xy_size - overlap))
            cut_x = xrange // x_num + overlap
            cut_y = yrange // y_num + overlap

            if verbose:
                print(f"n_tile: {x_num * y_num};",
                      f"\nx_slice_num: {x_num};", f"y_slice_num: {y_num};",
                      f"\nblock_x: {cut_x};", f"block_y: {cut_y};", f"overlap: {overlap};")
            
            with tqdm(total=x_num * y_num, desc='tile', disable=not verbose) as pbar:
                for x_pos in range(x_num):
                    pad_x = x_pos * (cut_x - overlap)
                    for y_pos in range(y_num):
                        pad_y = y_pos * (cut_y - overlap)
                        func_args = {
                            'pad_x': pad_x, 'cut_x': cut_x,
                            'pad_y': pad_y, 'cut_y': cut_y,
                            'x_pos': x_pos, 'y_pos': y_pos,
                            'x_num': x_num, 'y_num': y_num,
                            'overlap': overlap,
                            'data_dict': data_dict
                        }
                        func_args.update(kwargs)
                        func(*args, **func_args)
                        pbar.update(1)
        return wrapper
    return decorator
