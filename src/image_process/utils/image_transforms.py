"""
图像变换和处理工具函数
包含图像缩放、填充等操作
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.util import img_as_uint
from skimage.transform import resize
from skimage.morphology import white_tophat


def resize_pad(img, size):
    """
    将图像缩放到指定尺寸并填充到原始大小
    
    Args:
        img: 输入图像
        size: 目标尺寸 (height, width)
    
    Returns:
        处理后的图像
    """
    img_resized = resize(img, size, anti_aliasing=True)
    img_padded = np.zeros(img.shape)
    y_start, x_start = (img.shape[0] - size[0]) // 2, (img.shape[1] - size[1]) // 2
    img_padded[y_start:y_start+size[0], x_start:x_start+size[1]] = img_resized
    img_padded = img_as_uint(img_padded)
    return img_padded


def resize_dir(in_dir, out_dir, chn):
    """
    批量处理目录中的图像，按通道类型缩放
    
    Args:
        in_dir: 输入目录
        out_dir: 输出目录
        chn: 通道名称
    """
    Path(out_dir).mkdir(exist_ok=True)
    chn_sizes = {'cy3': 2302, 'TxRed': 2303, 'FAM': 2301, 'DAPI': 2300}
    size = chn_sizes[chn]
    im_list = list(Path(in_dir).glob(f'*.tif'))
    for im_path in tqdm(im_list, desc=Path(in_dir).name):
        im = imread(im_path)
        im = resize_pad(im, (size, size))
        imsave(Path(out_dir)/im_path.name, im, check_contrast=False)


def process_slice(slice_2d, channel):
    """
    处理单个切片，根据通道类型进行缩放
    
    Args:
        slice_2d: 2D图像切片
        channel: 通道名称
    
    Returns:
        处理后的切片
    """
    if channel != 'cy5':
        # resize and pad the slice
        chn_sizes = {'cy3': 2302, 'txred': 2303, 'fam': 2301, 'dapi': 2300}
        size = chn_sizes[channel]
        slice_2d = resize_pad(slice_2d, (size, size))
    return slice_2d


def create_ellipsoid_kernel(x_radius, y_radius, z_radius):
    """
    创建3D椭球形内核，用于形态学操作
    
    Args:
        x_radius, y_radius, z_radius: 各轴的半径
    
    Returns:
        椭球形内核
    """
    x = np.arange(-x_radius, x_radius+1)
    y = np.arange(-y_radius, y_radius+1)
    z = np.arange(-z_radius, z_radius+1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    ellipsoid = (xx**2 / x_radius**2) + (yy**2 / y_radius**2) + (zz**2 / z_radius**2)

    kernel = ellipsoid <= 1
    return kernel.astype(np.uint8)


def apply_tophat_filter(image, structure_element):
    """
    应用白顶帽滤波器
    
    Args:
        image: 输入图像
        structure_element: 结构元素
    
    Returns:
        滤波后的图像
    """
    return white_tophat(image, selem=structure_element)