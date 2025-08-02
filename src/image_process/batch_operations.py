"""
批量图像处理操作
包含批量缩放、3D堆栈创建等功能
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict
import glob
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from skimage.io import imread

from .utils.image_transforms import resize_dir, process_slice
from .os_snippets import try_mkdir


def resize_batch(in_dir, out_dir):
    """
    批量处理目录中的图像周期，按通道缩放
    
    Args:
        in_dir: 输入目录
        out_dir: 输出目录
    """
    try_mkdir(out_dir)
    cyc_paths = list(Path(in_dir).glob('cyc_*_*'))
    for cyc_path in cyc_paths:
        chn = cyc_path.name.split('_')[-1]
        if chn == 'cy5':
            shutil.copytree(cyc_path, Path(out_dir)/cyc_path.name)
        else:
            resize_dir(cyc_path, Path(out_dir)/cyc_path.name, chn)


def create_3d_stacks(src_dir, output_dir, channels, extract_points_cycle):
    """
    从2D切片创建3D堆栈
    
    Args:
        src_dir: 源目录
        output_dir: 输出目录
        channels: 要处理的通道列表
        extract_points_cycle: 要提取点的周期列表
    
    Returns:
        stack_name: 字典，包含每个瓦片的周期信息
        file_groups: 字典，包含文件分组信息
    """
    stack_name = dict()
    file_groups = defaultdict(list)
    
    for cyc_folder in glob.glob(os.path.join(src_dir, 'cy*')):
        for file_path in glob.glob(os.path.join(cyc_folder, '*.tif')):
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            cycle, tile, channel = parts[0], parts[1], parts[2]
            
            if channel in channels:
                z_index = int(filename.split('Z')[-1].split('.')[0])
                file_groups[(cycle, tile, channel)].append((z_index, file_path))
                if tile in stack_name: 
                    stack_name[tile].add(cycle)
                else: 
                    stack_name[tile] = set()
    
    stack_name = {key: sorted(list(value), key=lambda x: int(x[1:])) 
                  for key, value in stack_name.items()}
    file_groups = {k: sorted(v) for k, v in file_groups.items()}
    
    # 创建3D文件
    for (cycle, tile, channel), files in tqdm(file_groups.items(), desc='Processing stacks'):
        stack = np.array([process_slice(imread(file_path), channel) 
                         for _, file_path in files])
        os.makedirs(output_dir / tile / cycle, exist_ok=True)
        tifffile.imwrite(output_dir / tile / cycle / f"{channel.lower()}.tif", stack)
    
    return stack_name, file_groups


def process_spot_detection_batch(stack_name, air_dir, extract_points_cycle):
    """
    批量处理点检测
    
    Args:
        stack_name: 瓦片和周期信息字典
        air_dir: 处理目录
        extract_points_cycle: 要处理的周期列表
    """
    from ..AIRLOCALIZE.airlocalize import airlocalize
    
    for tile in tqdm(stack_name.keys(), desc='Detecting candidate points', position=0, leave=True):
        df = pd.DataFrame()
        df.to_csv(air_dir/tile/'combined_candidates.csv', index=False)
        
        for cycle in tqdm(extract_points_cycle, desc=f'Processing cycles for tile {tile}', 
                         position=1, leave=False):
            tile_cycle_dir = air_dir / tile / cycle
            
            # 执行airlocalization
            airlocalize(
                parameters_filename='/mnt/data/processing_codes/SPRINT_analysis/lib/AIRLOCALIZE/parameters.yaml', 
                default_parameters='/mnt/data/processing_codes/SPRINT_analysis/lib/AIRLOCALIZE/parameters_default.yaml',
                update={'dataFileName': tile_cycle_dir, 'saveDirName': tile_cycle_dir, 'verbose': False}
            )
            
            spots_file = [_ for _ in os.listdir(tile_cycle_dir) if _.endswith('spots.csv')]
            df = pd.read_csv(air_dir / tile / 'combined_candidates.csv')
            
            if len(df) > 0: 
                df = pd.concat([df] + [pd.read_csv(tile_cycle_dir / file) for file in spots_file], axis=0)
            else: 
                df = pd.concat([pd.read_csv(tile_cycle_dir / file) for file in spots_file], axis=0)
            df.to_csv(air_dir/tile/'combined_candidates.csv', index=False)