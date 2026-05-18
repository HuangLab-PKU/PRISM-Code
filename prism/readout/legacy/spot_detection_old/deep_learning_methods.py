"""
深度学习方法实现
Deep learning methods implementation

集成现有的 StarDist 预测和高斯拟合功能
Integrates existing StarDist prediction and Gaussian fitting functionality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from tqdm import tqdm

from .spot_detection import (
    CoordinateDetector, IntensityExtractor, SpotDetectionResult, ImageProcessor
)
from .gaussian_fitting import fit_gaussian_2d, get_intensity_and_background

logger = logging.getLogger(__name__)

# 延迟导入以避免依赖问题
try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    from scipy.ndimage import center_of_mass
    STAR_DIST_AVAILABLE = True
except ImportError:
    STAR_DIST_AVAILABLE = False
    logger.warning("StarDist not available. Deep learning methods will be disabled.")


class StarDistDetector(CoordinateDetector):
    """StarDist坐标检测器"""
    
    def __init__(self, model_path: str, model_name: str,
                 prob_thresh: float = 0.5, nms_thresh: float = 0.3):
        if not STAR_DIST_AVAILABLE:
            raise ImportError("StarDist is not available. Please install stardist.")
        
        self.model_path = model_path
        self.model_name = model_name
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self._model = None
    
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._model = StarDist2D(None, name=self.model_name, basedir=self.model_path)
        return self._model
    
    def detect_coordinates(self, image: np.ndarray) -> np.ndarray:
        """使用StarDist检测信号点坐标"""
        # 确保图像是2D的
        if len(image.shape) == 3:
            # 如果是多通道，取第一个通道或合并通道
            if image.shape[-1] <= 4:  # 多通道图像
                image = np.mean(image, axis=-1)
            else:  # 3D图像，取中间层
                image = image[image.shape[0] // 2]
        
        # 归一化图像
        img_normalized = normalize(image, 1, 99.8, axis=(0, 1))
        
        # 预测
        labels, _ = self.model.predict_instances_big(
            img_normalized,
            prob_thresh=self.prob_thresh,
            nms_thresh=self.nms_thresh
        )
        
        # 计算质心
        return self._get_spot_centroids(labels)
    
    def _get_spot_centroids(self, labels: np.ndarray) -> np.ndarray:
        """计算标签图像的质心坐标"""
        if labels.max() == 0:
            return np.empty((0, 2), dtype=int)
        
        centroids = center_of_mass(labels, labels, index=np.arange(1, labels.max() + 1))
        return np.array(centroids, dtype=int)


class GaussianIntensityExtractor(IntensityExtractor):
    """高斯拟合强度提取器"""
    
    def __init__(self, roi_size: int = 15):
        self.roi_size = roi_size
    
    def extract_intensities(self, image: np.ndarray, 
                          coordinates: np.ndarray) -> Dict[str, np.ndarray]:
        """使用高斯拟合提取强度"""
        if len(coordinates) == 0:
            return {
                'intensity': np.array([]),
                'background': np.array([]),
                'amplitude': np.array([]),
                'sigma_x': np.array([]),
                'sigma_y': np.array([])
            }
        
        intensities = []
        backgrounds = []
        amplitudes = []
        sigma_xs = []
        sigma_ys = []
        
        for y, x in tqdm(coordinates, desc="Gaussian fitting", leave=False):
            popt, _ = fit_gaussian_2d(image, y, x, roi_size=self.roi_size)
            
            if popt is not None:
                intensity, background = get_intensity_and_background(popt)
                amplitude, _, _, sigma_x, sigma_y, _, _ = popt
                
                intensities.append(intensity)
                backgrounds.append(background)
                amplitudes.append(amplitude)
                sigma_xs.append(sigma_x)
                sigma_ys.append(sigma_y)
            else:
                # 拟合失败，使用直接读取
                intensities.append(image[y, x])
                backgrounds.append(0)
                amplitudes.append(image[y, x])
                sigma_xs.append(1.0)
                sigma_ys.append(1.0)
        
        return {
            'intensity': np.array(intensities),
            'background': np.array(backgrounds),
            'amplitude': np.array(amplitudes),
            'sigma_x': np.array(sigma_xs),
            'sigma_y': np.array(sigma_ys)
        }


class MaskIntensityExtractor(IntensityExtractor):
    """掩码区域强度提取器"""
    
    def __init__(self, mask_radius: int = 3, background_radius: int = 8):
        self.mask_radius = mask_radius
        self.background_radius = background_radius
    
    def extract_intensities(self, image: np.ndarray, 
                          coordinates: np.ndarray) -> Dict[str, np.ndarray]:
        """使用掩码区域提取强度"""
        if len(coordinates) == 0:
            return {
                'intensity': np.array([]),
                'background': np.array([]),
                'snr': np.array([])
            }
        
        intensities = []
        backgrounds = []
        snrs = []
        
        for y, x in coordinates:
            # 创建信号掩码
            y_min = max(0, y - self.mask_radius)
            y_max = min(image.shape[0], y + self.mask_radius + 1)
            x_min = max(0, x - self.mask_radius)
            x_max = min(image.shape[1], x + self.mask_radius + 1)
            
            signal_mask = np.zeros_like(image, dtype=bool)
            signal_mask[y_min:y_max, x_min:x_max] = True
            
            # 创建背景掩码
            bg_y_min = max(0, y - self.background_radius)
            bg_y_max = min(image.shape[0], y + self.background_radius + 1)
            bg_x_min = max(0, x - self.background_radius)
            bg_x_max = min(image.shape[1], x + self.background_radius + 1)
            
            background_mask = np.zeros_like(image, dtype=bool)
            background_mask[bg_y_min:bg_y_max, bg_x_min:bg_x_max] = True
            background_mask[signal_mask] = False  # 排除信号区域
            
            # 计算强度
            signal_intensity = np.sum(image[signal_mask])
            background_intensity = np.mean(image[background_mask]) if np.any(background_mask) else 0
            snr = signal_intensity / background_intensity if background_intensity > 0 else float('inf')
            
            intensities.append(signal_intensity)
            backgrounds.append(background_intensity)
            snrs.append(snr)
        
        return {
            'intensity': np.array(intensities),
            'background': np.array(backgrounds),
            'snr': np.array(snrs)
        }


class DeepLearningSpotDetector:
    """深度学习信号点检测器"""
    
    def __init__(self, 
                 model_path: str,
                 model_name: str,
                 prob_thresh: float = 0.5,
                 nms_thresh: float = 0.3,
                 roi_size: int = 15,
                 intensity_method: str = 'gaussian'):
        """
        初始化深度学习检测器
        
        Args:
            model_path: 模型路径
            model_name: 模型名称
            prob_thresh: 概率阈值
            nms_thresh: NMS阈值
            roi_size: 高斯拟合ROI大小
            intensity_method: 强度提取方法 ('gaussian', 'mask', 'direct')
        """
        self.coordinate_detector = StarDistDetector(
            model_path, model_name, prob_thresh, nms_thresh
        )
        
        if intensity_method == 'gaussian':
            self.intensity_extractor = GaussianIntensityExtractor(roi_size)
        elif intensity_method == 'mask':
            self.intensity_extractor = MaskIntensityExtractor()
        else:
            from .spot_detection import DirectIntensityExtractor
            self.intensity_extractor = DirectIntensityExtractor()
        
        self.image_processor = ImageProcessor()
    
    def detect_spots(self, image: np.ndarray) -> SpotDetectionResult:
        """检测信号点"""
        # 1. 坐标检测
        coordinates = self.coordinate_detector.detect_coordinates(image)
        
        # 2. 强度提取
        intensities = self.intensity_extractor.extract_intensities(image, coordinates)
        
        # 3. 构建结果
        metadata = {
            'image_shape': image.shape,
            'num_spots': len(coordinates),
            'model_path': self.coordinate_detector.model_path,
            'model_name': self.coordinate_detector.model_name,
            'prob_thresh': self.coordinate_detector.prob_thresh,
            'nms_thresh': self.coordinate_detector.nms_thresh
        }
        
        processing_info = {
            'method': 'deep_learning',
            'intensity_method': type(self.intensity_extractor).__name__
        }
        
        return SpotDetectionResult(
            coordinates=coordinates,
            intensities=intensities,
            metadata=metadata,
            processing_info=processing_info
        )


class MultiChannelDeepLearningDetector:
    """多通道深度学习检测器"""
    
    def __init__(self, 
                 model_path: str,
                 model_name: str,
                 prob_thresh: float = 0.5,
                 nms_thresh: float = 0.3,
                 roi_size: int = 15,
                 intensity_method: str = 'gaussian'):
        
        self.detector = DeepLearningSpotDetector(
            model_path, model_name, prob_thresh, nms_thresh, roi_size, intensity_method
        )
        self.image_processor = ImageProcessor()
    
    def detect_spots_multi_channel(self, 
                                 image_dict: Dict[str, np.ndarray],
                                 use_combined_detection: bool = True) -> SpotDetectionResult:
        """
        多通道深度学习检测
        
        Args:
            image_dict: 通道名称到图像的映射
            use_combined_detection: 是否使用合并图像进行检测
            
        Returns:
            检测结果
        """
        if use_combined_detection and len(image_dict) > 1:
            # 合并所有通道进行检测
            images = list(image_dict.values())
            combined_image = np.stack(images, axis=-1)
            
            # 检测坐标
            result = self.detector.detect_spots(combined_image)
            coordinates = result.coordinates
        else:
            # 使用第一个通道检测
            first_channel = list(image_dict.keys())[0]
            first_image = image_dict[first_channel]
            result = self.detector.detect_spots(first_image)
            coordinates = result.coordinates
        
        # 为每个通道提取强度
        all_intensities = {}
        for channel, image in image_dict.items():
            logger.info(f"Extracting intensities for channel: {channel}")
            
            intensities = self.detector.intensity_extractor.extract_intensities(
                image, coordinates
            )
            
            for intensity_key, intensity_values in intensities.items():
                key = f"{channel}_{intensity_key}"
                all_intensities[key] = intensity_values
        
        # 构建最终结果
        metadata = {
            'channels': list(image_dict.keys()),
            'num_spots': len(coordinates),
            'use_combined_detection': use_combined_detection,
            'model_path': self.detector.coordinate_detector.model_path,
            'model_name': self.detector.coordinate_detector.model_name
        }
        
        processing_info = {
            'method': 'multi_channel_deep_learning',
            'num_channels': len(image_dict),
            'intensity_method': type(self.detector.intensity_extractor).__name__
        }
        
        return SpotDetectionResult(
            coordinates=coordinates,
            intensities=all_intensities,
            metadata=metadata,
            processing_info=processing_info
        )


def create_deep_learning_detector(model_path: str, model_name: str, **kwargs) -> DeepLearningSpotDetector:
    """创建深度学习检测器的工厂函数"""
    return DeepLearningSpotDetector(model_path, model_name, **kwargs)


def create_multi_channel_deep_learning_detector(model_path: str, model_name: str, **kwargs) -> MultiChannelDeepLearningDetector:
    """创建多通道深度学习检测器的工厂函数"""
    return MultiChannelDeepLearningDetector(model_path, model_name, **kwargs)


