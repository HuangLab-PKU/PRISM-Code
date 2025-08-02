"""
坐标处理和拼接功能
包含位移校正和3D拼接操作
"""
import pandas as pd
import numpy as np


def shift_correction(signal_df, shift_df, tile, cyc, ref_cyc=1):
    """
    应用位移校正到信号坐标
    
    Args:
        signal_df: 包含坐标信息的数据框
        shift_df: 位移信息数据框
        tile: 瓦片编号
        cyc: 当前周期
        ref_cyc: 参考周期
    
    Returns:
        校正后的坐标数据框
    """
    file = f'FocalStack_{tile:03d}.tif'
    signal_df = signal_df.copy()
    
    # Handle the case where the index might not exist
    try:
        if cyc != ref_cyc:
            shift_entry = shift_df.loc[(cyc, file)]
            y_shift, x_shift = map(int, shift_entry.split(' '))
            signal_df.loc[:, 'x_in_pix'] += x_shift
            signal_df.loc[:, 'y_in_pix'] += y_shift
    except KeyError:
        print(f"Cycle {cyc} or file {file} not found in shift_df.")
        # Optionally, handle the absence of the key more gracefully here
        pass
    
    return signal_df


def stitch_3d(signal_df, meta_df, tile):
    """
    执行3D坐标拼接，将局部坐标转换为全局坐标
    
    Args:
        signal_df: 包含局部坐标的信号数据框
        meta_df: 包含瓦片位置信息的元数据
        tile: 瓦片编号
    
    Returns:
        包含全局坐标的数据框
    """
    file = f'FocalStack_{tile:03d}.tif'
    # Get the metadata row for this file to get its global position
    meta_row = meta_df.loc[file]
    # Get global positions
    global_x_start, global_y_start = meta_row['x'], meta_row['y']
    # Vectorized calculation to update signal_df directly
    signal_df['y_in_pix'] = signal_df['y_in_pix'] + global_y_start
    signal_df['x_in_pix'] = signal_df['x_in_pix'] + global_x_start
    
    return signal_df


def filter_coordinates_by_bounds(coordinates, shape):
    """
    根据图像边界过滤坐标
    
    Args:
        coordinates: 坐标数据框
        shape: 图像形状 (z, y, x)
    
    Returns:
        过滤后的坐标
    """
    return coordinates[
        (0 <= coordinates['z_in_pix']) & (coordinates['z_in_pix'] < shape[0]) &
        (0 <= coordinates['y_in_pix']) & (coordinates['y_in_pix'] < shape[1]) &
        (0 <= coordinates['x_in_pix']) & (coordinates['x_in_pix'] < shape[2])
    ]


def extract_intensity_at_coordinates(image, coordinates):
    """
    在指定坐标位置提取图像强度值
    
    Args:
        image: 3D图像数组
        coordinates: 坐标数据框，包含z_in_pix, y_in_pix, x_in_pix列
    
    Returns:
        强度值数组
    """
    z_coords = coordinates['z_in_pix'].to_numpy()
    y_coords = coordinates['y_in_pix'].to_numpy()
    x_coords = coordinates['x_in_pix'].to_numpy()
    
    return image[z_coords, y_coords, x_coords]