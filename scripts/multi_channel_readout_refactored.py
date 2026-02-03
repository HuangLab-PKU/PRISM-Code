"""
重构后的多通道读取脚本
Refactored multi-channel readout script
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
import tifffile
from pprint import pprint

# 导入重构后的模块
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from spot_detection import (
    tophat_spots, extract_coordinates, calculate_snr, extract_signal,
    read_intensity, remove_duplicates, divide_main,
    load_config, merge_configs
)
from spot_detection.channel_transformation import (
    apply_transformation_matrix, validate_transformation_matrix, print_transformation_matrix
)


class MultiChannelProcessor:
    """多通道处理器类"""
    
    def __init__(self, config_path=None, run_id=None, prism_panel=None):
        """
        初始化处理器
        
        Args:
            config_path: 配置文件路径
            run_id: 运行ID
            prism_panel: PRISM面板类型
        """
        # 加载配置
        self.config = load_config(config_path)
        
        # 设置运行参数
        self.run_id = run_id or 'Example_dataset'
        self.prism_panel = prism_panel or self.config['base']['prism_panel']
        
        # 设置路径
        self.base_dir = Path(self.config['base']['base_dir'])
        self.src_dir = self.base_dir / f'{self.run_id}'
        self.stc_dir = self.src_dir / 'stitched'
        self.read_dir = self.src_dir / 'readout'
        self.tmp_dir = self.read_dir / 'tmp'
        
        # 创建目录
        self.read_dir.mkdir(exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)
        
        # 获取配置参数
        self.channels = self.config['base']['channels']
        self.max_memory = self.config['base']['max_memory']
        
    def process_single_run(self):
        """处理单个运行"""
        print(f'处理RUN_ID: {self.run_id}')
        print(f'PRISM面板: {self.prism_panel}')
        
        # 获取图像形状
        with tifffile.TiffFile(self.stc_dir / f'cyc_1_{self.channels[0]}.tif') as tif:
            image_shape = tif.series[0].shape
            
        if (image_shape[0] == 1 and len(image_shape) == 3) or len(image_shape) < 3:
            if len(image_shape) == 3:
                image_shape = image_shape[1:]
            print('2D_image_shape_y, x: ({})'.format(', '.join(map(str, image_shape))))
        else:
            print('3D_image_shape_z, x, y: ({})'.format(', '.join(map(str, image_shape))))

        # 计算最大体积
        file_size_bytes = os.path.getsize(self.stc_dir / f'cyc_1_{self.channels[0]}.tif')
        file_size_gb = file_size_bytes / (1024 ** 3)
        print(f'图像文件大小: {file_size_bytes} bytes ({file_size_gb:.2f} GB)')
        
        max_volume = self.max_memory * 1024 ** 3  # byte
        max_volume /= 2  # 2byte for one pixel
        max_volume /= len(self.channels)  # 多通道
        max_volume /= 2  # tophat to double
        max_volume /= 2  # more space

        # 提取强度
        tophat_mean_dict = self._extract_tophat_means(image_shape)
        
        # 处理图像块
        self._process_image_blocks(image_shape, max_volume, tophat_mean_dict)
        
        # 拼接所有块
        intensity = self._stitch_blocks(image_shape, max_volume)
        
        # 后处理
        intensity = self._post_process(intensity)
        
        # 保存结果
        self._save_results(intensity)
        
        # 清理临时文件
        self._cleanup()
        
        return intensity
    
    def _extract_tophat_means(self, image_shape):
        """提取tophat均值"""
        tophat_mean_dict = {}
        memmap_shape = None
        memmap_dtype = None
        
        for channel in self.channels:
            with tifffile.TiffFile(self.stc_dir / f'cyc_1_{channel}.tif') as tif:
                image = tif.asarray()
                tophat_mean_dict[channel] = np.mean(tophat_spots(image, self.config['image_processing']['tophat_kernel_size']))
                
                if memmap_shape is None:
                    memmap_shape = image.shape
                    memmap_dtype = image.dtype
                
                # 创建内存映射文件
                if f'cyc_1_{channel}.dat' not in os.listdir(self.tmp_dir):
                    memmap_array = np.memmap(
                        self.tmp_dir / f'cyc_1_{channel}.dat', 
                        dtype=memmap_dtype, 
                        mode='w+', 
                        shape=memmap_shape
                    )
                    memmap_array[:] = image[:]
                    memmap_array.flush()
                    del memmap_array

        print('tophat_mean:')
        pprint(tophat_mean_dict)
        
        # 保存tophat均值
        tophat_mean_dict = {k: float(v) for k, v in tophat_mean_dict.items()}
        with open(self.tmp_dir / 'tophat_mean.yaml', 'w') as f:
            yaml.dump({'tophat_mean': tophat_mean_dict}, f)
        
        return tophat_mean_dict
    
    def _process_image_blocks(self, image_shape, max_volume, tophat_mean_dict):
        """处理图像块"""
        @divide_main(shape=image_shape, max_volume=max_volume, overlap=self.config['batch']['overlap'])
        def extract_intensity(**kwargs):
            pad_x = kwargs['pad_x']
            cut_x = kwargs['cut_x']
            pad_y = kwargs['pad_y']
            cut_y = kwargs['cut_y']
            x_pos = kwargs['x_pos']
            y_pos = kwargs['y_pos']

            image_dict = {}
            coordinate_dict = {}
            snr_dict = {}

            def process_channel(channel):
                image_path = self.tmp_dir / f'cyc_1_{channel}.dat'
                image = np.memmap(str(image_path), dtype=memmap_dtype, mode='r', shape=memmap_shape)
                
                coordinate, snr, image_data = extract_signal(
                    image, pad_x, cut_x, pad_y, cut_y,
                    snr=self.config['signal_processing']['snr'],
                    tophat_mean=tophat_mean_dict[channel],
                    abs_thre=self.config['image_processing']['local_max_abs_thre_ch'][channel],
                    tophat_break=self.config['image_processing']['tophat_break'],
                    intensity_thre=self.config['image_processing']['intensity_thre'],
                    check_snr=self.config['image_processing']['cal_snr'],
                    kernel_size=self.config['signal_processing']['kernel_size']
                )
                return channel, coordinate, snr, image_data
            
            with Pool() as pool:
                results = pool.map(process_channel, self.channels)

            for channel, coordinate, snr, image_data in results:
                coordinate_dict[channel] = coordinate
                image_dict[channel] = image_data
                if self.config['image_processing']['cal_snr']:
                    snr_dict[channel] = snr

            if self.config['image_processing']['cal_snr']:
                intensity = pd.concat([
                    read_intensity(image_dict, coordinate_dict[channel], channel=channel, snr=snr_dict[channel]) 
                    for channel in coordinate_dict.keys()
                ])
            else:
                intensity = pd.concat([
                    read_intensity(image_dict, coordinate_dict[channel], channel=channel) 
                    for channel in coordinate_dict.keys()
                ])
            
            intensity['X'] = intensity['X'] + pad_x
            intensity['Y'] = intensity['Y'] + pad_y
            intensity.to_csv(self.tmp_dir / f'intensity_block_{x_pos}_{y_pos}.csv')

        extract_intensity()
    
    def _stitch_blocks(self, image_shape, max_volume):
        """拼接所有块"""
        global intensity
        intensity = pd.DataFrame()

        @divide_main(shape=image_shape, max_volume=max_volume, overlap=self.config['batch']['overlap'], verbose=False)
        def stitch_all_block(**kwargs):
            global intensity
            pad_x = kwargs['pad_x']
            cut_x = kwargs['cut_x']
            pad_y = kwargs['pad_y']
            cut_y = kwargs['cut_y']
            x_pos = kwargs['x_pos']
            y_pos = kwargs['y_pos']
            x_num = kwargs['x_num']
            y_num = kwargs['y_num']
            overlap = kwargs['overlap']

            tmp_intensity = pd.read_csv(self.tmp_dir / f'intensity_block_{x_pos}_{y_pos}.csv', index_col=0)
            
            xmin, xmax = pad_x + overlap//4, pad_x + cut_x - overlap//4
            if x_pos == 1:
                xmin = 0
            elif x_pos == x_num:
                xmax = pad_x + cut_x

            ymin, ymax = pad_y + overlap//4, pad_y + cut_y - overlap//4
            if y_pos == 1:
                ymin = 0
            elif y_pos == y_num:
                ymax = pad_y + cut_y

            tmp_intensity = tmp_intensity[
                (tmp_intensity['Y'] >= ymin) & (tmp_intensity['Y'] <= ymax) &
                (tmp_intensity['X'] >= xmin) & (tmp_intensity['X'] <= xmax)
            ]
            
            intensity = pd.concat([intensity, tmp_intensity])

        stitch_all_block()
        return intensity
    
    def _post_process(self, intensity):
        """后处理"""
        # 重命名列
        channel_mapping = self.config['channel_mapping']
        intensity = intensity.rename(columns=channel_mapping)
        intensity = intensity.reset_index(drop=True)
        intensity.to_csv(self.tmp_dir / 'intensity_raw.csv')
        print('原始读取:', len(intensity))

        # 使用转换矩阵进行通道矫正
        transformation_matrix = self.config['channel_transformation_matrix']
        
        # 验证转换矩阵
        if not validate_transformation_matrix(transformation_matrix):
            raise ValueError("转换矩阵验证失败")
        
        # 打印转换矩阵信息（仅在调试模式下）
        if hasattr(self, 'debug') and self.debug:
            print_transformation_matrix(transformation_matrix)
        
        # 应用转换矩阵
        intensity = apply_transformation_matrix(intensity, transformation_matrix)

        # 计算总和和归一化
        intensity['sum'] = intensity['Scaled_R'] + intensity['Scaled_Ye'] + intensity['Scaled_B']
        intensity['G/A'] = intensity['Scaled_G'] / intensity['sum']

        # 应用阈值
        thresholds = self.config['thresholds']
        if self.prism_panel in ('PRISM31', 'PRISM46', 'PRISM64'):
            intensity = intensity[
                (intensity['sum'] >= thresholds['sum_threshold']) | 
                ((intensity['G/A'] >= thresholds['g_threshold']) & 
                 (intensity['Scaled_G'] > thresholds['g_abs_threshold']))
            ]
        elif self.prism_panel in ('PRISM30', 'PRISM45', 'PRISM63'):
            intensity = intensity[intensity['sum'] >= thresholds['sum_threshold']]
            intensity = intensity.dropna()
        
        intensity.loc[intensity['G/A'] > thresholds['g_maxvalue'], 'G/A'] = thresholds['g_maxvalue']
        print('去除低强度:', len(intensity))

        # 去重
        intensity['ID'] = intensity['Y'] * 10**7 + intensity['X']
        intensity = intensity.drop_duplicates(subset=['Y', 'X'])

        df = intensity[['Y', 'X', 'R', 'Ye', 'B', 'G']]
        coordinates = df[['Y', 'X']].values
        coordinates = remove_duplicates(coordinates)
        df_reduced = pd.DataFrame(coordinates, columns=['Y', 'X'])

        df_reduced['ID'] = df_reduced['Y'] * 10**7 + df_reduced['X']
        intensity = intensity[intensity['ID'].isin(df_reduced['ID'])]
        intensity = intensity.drop(columns=['ID'])
        print('去重后:', len(intensity))

        return intensity
    
    def _save_results(self, intensity):
        """保存结果"""
        intensity.to_csv(self.read_dir / 'intensity_deduplicated.csv')
        
        # 复制当前脚本到结果目录
        current_file_path = os.path.abspath(__file__)
        target_file_path = os.path.join(self.read_dir, os.path.basename(current_file_path))
        try:
            shutil.copy(current_file_path, target_file_path)
        except shutil.SameFileError:
            print('文件已存在于目标目录中。')
        except Exception as e:
            print(f'复制文件时出错: {e}')
    
    def _cleanup(self):
        """清理临时文件"""
        for file_path in glob.glob(str(self.tmp_dir / '*.dat')):
            os.remove(file_path)
        print("已删除.dat文件")


def main(config_path=None, run_id=None, prism_panel=None):
    """主函数"""
    processor = MultiChannelProcessor(config_path, run_id, prism_panel)
    return processor.process_single_run()


if __name__ == '__main__':
    # 示例用法
    main()
