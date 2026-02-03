"""
通道转换矩阵工具
Channel transformation matrix utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def apply_transformation_matrix(intensity_df: pd.DataFrame, 
                              transformation_matrix: Dict[str, List[float]], 
                              input_channels: List[str] = None,
                              output_channels: List[str] = None) -> pd.DataFrame:
    """
    应用通道转换矩阵到强度数据
    
    Args:
        intensity_df: 包含原始通道强度的DataFrame
        transformation_matrix: 转换矩阵字典
        input_channels: 输入通道列表，默认为['R', 'Ye', 'G', 'B']
        output_channels: 输出通道列表，默认为['R', 'Ye', 'G', 'B']
        
    Returns:
        包含矫正后强度的DataFrame
    """
    if input_channels is None:
        input_channels = ['R', 'Ye', 'G', 'B']
    if output_channels is None:
        output_channels = ['R', 'Ye', 'G', 'B']
    
    # 构建输入向量
    raw_intensities = intensity_df[input_channels].values  # shape: (n_spots, 4)
    
    # 构建转换矩阵
    matrix = np.array([transformation_matrix[channel] for channel in output_channels])
    
    # 应用矩阵变换: output = matrix @ input.T
    corrected_intensities = np.dot(matrix, raw_intensities.T).T  # shape: (n_spots, 4)
    
    # 确保B通道非负（串扰消除后可能为负）
    if 'B' in output_channels:
        b_idx = output_channels.index('B')
        corrected_intensities[:, b_idx] = np.maximum(corrected_intensities[:, b_idx], 0)
    
    # 创建结果DataFrame
    result_df = intensity_df.copy()
    for i, channel in enumerate(output_channels):
        result_df[f'Scaled_{channel}'] = corrected_intensities[:, i]
    
    return result_df


def validate_transformation_matrix(transformation_matrix: Dict[str, List[float]]) -> bool:
    """
    验证转换矩阵的格式和合理性
    
    Args:
        transformation_matrix: 转换矩阵字典
        
    Returns:
        是否有效
    """
    try:
        # 检查所有必需的通道
        required_channels = ['R', 'Ye', 'G', 'B']
        for channel in required_channels:
            if channel not in transformation_matrix:
                print(f"错误: 缺少通道 {channel} 的转换矩阵")
                return False
        
        # 检查矩阵维度
        matrix_size = len(transformation_matrix['R'])
        for channel in required_channels:
            if len(transformation_matrix[channel]) != matrix_size:
                print(f"错误: 通道 {channel} 的矩阵维度不一致")
                return False
        
        if matrix_size != 4:
            print(f"错误: 矩阵维度应为4x4，实际为{len(required_channels)}x{matrix_size}")
            return False
        
        # 检查矩阵是否为数值
        matrix = np.array([transformation_matrix[channel] for channel in required_channels])
        if not np.isfinite(matrix).all():
            print("错误: 矩阵包含非有限数值")
            return False
        
        print("✓ 转换矩阵验证通过")
        return True
        
    except Exception as e:
        print(f"错误: 转换矩阵验证失败 - {e}")
        return False


def print_transformation_matrix(transformation_matrix: Dict[str, List[float]], 
                              input_channels: List[str] = None,
                              output_channels: List[str] = None):
    """
    打印转换矩阵的详细信息
    
    Args:
        transformation_matrix: 转换矩阵字典
        input_channels: 输入通道列表
        output_channels: 输出通道列表
    """
    if input_channels is None:
        input_channels = ['R', 'Ye', 'G', 'B']
    if output_channels is None:
        output_channels = ['R', 'Ye', 'G', 'B']
    
    print("通道转换矩阵:")
    print("=" * 60)
    print("格式: [输出通道] = 矩阵行 × [输入通道]")
    print()
    
    # 打印表头
    header = "输出\\输入".ljust(12)
    for input_ch in input_channels:
        header += input_ch.rjust(8)
    print(header)
    print("-" * len(header))
    
    # 打印矩阵行
    for output_ch in output_channels:
        row_str = output_ch.ljust(12)
        for i, input_ch in enumerate(input_channels):
            value = transformation_matrix[output_ch][i]
            row_str += f"{value:8.3f}"
        print(row_str)
    
    print()
    print("数学表达式:")
    for output_ch in output_channels:
        terms = []
        for i, input_ch in enumerate(input_channels):
            coeff = transformation_matrix[output_ch][i]
            if coeff != 0:
                if coeff == 1:
                    terms.append(f"{input_ch}")
                elif coeff == -1:
                    terms.append(f"-{input_ch}")
                else:
                    terms.append(f"{coeff:.3f}×{input_ch}")
        
        if not terms:
            expression = "0"
        else:
            expression = " + ".join(terms).replace(" + -", " - ")
        
        print(f"{output_ch}_corrected = {expression}")


def create_identity_matrix() -> Dict[str, List[float]]:
    """创建单位矩阵（无转换）"""
    return {
        'R': [1.0, 0.0, 0.0, 0.0],
        'Ye': [0.0, 1.0, 0.0, 0.0],
        'G': [0.0, 0.0, 1.0, 0.0],
        'B': [0.0, 0.0, 0.0, 1.0]
    }


def create_scaling_matrix(scaling_factors: Dict[str, float]) -> Dict[str, List[float]]:
    """创建仅包含缩放的转换矩阵"""
    matrix = create_identity_matrix()
    for channel, factor in scaling_factors.items():
        if channel in matrix:
            channel_idx = ['R', 'Ye', 'G', 'B'].index(channel)
            matrix[channel][channel_idx] = factor
    return matrix


def create_crosstalk_matrix(crosstalk_factors: Dict[str, float]) -> Dict[str, List[float]]:
    """
    创建串扰消除矩阵
    
    Args:
        crosstalk_factors: 串扰因子字典，格式如 {'B_from_G': 0.25}
    """
    matrix = create_identity_matrix()
    
    for key, factor in crosstalk_factors.items():
        if '_from_' in key:
            target_channel, source_channel = key.split('_from_')
            if target_channel in matrix and source_channel in ['R', 'Ye', 'G', 'B']:
                target_idx = ['R', 'Ye', 'G', 'B'].index(target_channel)
                source_idx = ['R', 'Ye', 'G', 'B'].index(source_channel)
                matrix[target_channel][source_idx] = -factor  # 负号表示消除串扰
    
    return matrix

