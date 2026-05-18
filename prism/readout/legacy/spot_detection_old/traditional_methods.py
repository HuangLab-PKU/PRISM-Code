"""
传统图像处理方法实现
Traditional image processing methods implementation

集成现有的 multi_channel_readout 相关函数
Integrates existing multi_channel_readout related functions
"""

import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial import KDTree
from skimage.feature import peak_local_max
from tqdm import tqdm
import logging

from .spot_detection import (
    BackgroundRemover,
    CoordinateDetector,
    IntensityExtractor,
    SpotDetectionResult,
    ImageProcessor,
)

logger = logging.getLogger(__name__)


class TophatBackgroundRemover(BackgroundRemover):
    """Tophat背景去除器 - 基于现有实现"""

    def __init__(self, kernel_size: int = 7, tophat_break: float = 100):
        self.kernel_size = kernel_size
        self.tophat_break = tophat_break

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """使用Tophat形态学操作去除背景"""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        tophat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

        # 应用阈值
        tophat_image[tophat_image < self.tophat_break] = 0

        return tophat_image


class LocalMaximaDetector(CoordinateDetector):
    """局部最大值检测器 - 基于现有 extract_coordinates 实现"""

    def __init__(
        self,
        min_distance: int = 2,
        local_max_thre: float = 200,
        intensity_thre: Optional[float] = None,
    ):
        self.min_distance = min_distance
        self.local_max_thre = local_max_thre
        self.intensity_thre = intensity_thre

    def detect_coordinates(self, image: np.ndarray) -> np.ndarray:
        """检测局部最大值坐标"""
        coordinates = peak_local_max(
            image, min_distance=self.min_distance, threshold_abs=self.local_max_thre
        )

        # 应用强度阈值过滤
        if self.intensity_thre is not None:
            if self.intensity_thre <= 1:  # 分位数阈值
                intensities = image[coordinates[:, 0], coordinates[:, 1]]
                threshold_value = np.quantile(intensities, self.intensity_thre)
            else:  # 绝对阈值
                threshold_value = self.intensity_thre

            coordinates = coordinates[
                image[coordinates[:, 0], coordinates[:, 1]] > threshold_value
            ]

        return np.array(coordinates)


class DirectIntensityExtractor(IntensityExtractor):
    """直接强度提取器 - 基于现有 read_intensity 实现"""

    def __init__(self, kernel_size: int = 5, dilation_iterations: int = 1):
        self.kernel_size = kernel_size
        self.dilation_iterations = dilation_iterations

    def extract_intensities(
        self, image: np.ndarray, coordinates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """直接读取坐标位置的强度值"""
        if len(coordinates) == 0:
            return {"intensity": np.array([])}

        # 创建掩码
        mask = np.zeros(image.shape, dtype=np.uint16)
        mask[coordinates[:, 0], coordinates[:, 1]] = 255

        # 应用掩码
        masked_image = image.copy()
        masked_image[mask <= 0] = 0

        # 膨胀操作
        if self.dilation_iterations > 0:
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            masked_image = cv2.dilate(
                masked_image, kernel, iterations=self.dilation_iterations
            )

        # 提取强度
        intensities = masked_image[coordinates[:, 0], coordinates[:, 1]]

        return {"intensity": intensities}


class SNRCalculator:
    """信噪比计算器 - 基于现有 calculate_snr 实现"""

    def __init__(self, neighborhood_size: int = 10):
        self.neighborhood_size = neighborhood_size

    def calculate_snr(self, image: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """计算给定点的信噪比"""
        offset = -int(-self.neighborhood_size // 2)
        padded_img = np.pad(image, offset, mode="edge")
        snr_values = []

        for point in tqdm(coordinates, desc="Calculating SNR", leave=False):
            y, x = point
            x_min, x_max = x, x + self.neighborhood_size
            y_min, y_max = y, y + self.neighborhood_size

            neighborhood = padded_img[y_min:y_max, x_min:x_max]
            min_val = np.min(neighborhood)
            point_val = image[y, x]
            snr = point_val / min_val if min_val != 0 else float("inf")
            snr_values.append(snr)

        return np.array(snr_values)


class DuplicateRemover:
    """重复点移除器 - 基于现有 remove_duplicates 实现"""

    def __init__(self, distance_threshold: float = 2.0):
        self.distance_threshold = distance_threshold

    def remove_duplicates(self, coordinates: np.ndarray) -> np.ndarray:
        """移除重复的坐标点"""
        if len(coordinates) == 0:
            return coordinates

        tree = KDTree(coordinates)
        pairs = tree.query_pairs(self.distance_threshold)

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

        return np.delete(coordinates, list(discard), axis=0)


class TraditionalSpotDetector:
    """传统信号点检测器 - 支持配置化的方法选择"""

    def __init__(
        self,
        background_removal: Optional[Dict] = None,
        coordinate_detection: Optional[Dict] = None,
        snr_threshold: float = 8.0,
        check_snr: bool = False,
        remove_duplicates: bool = True,
        distance_threshold: float = 2.0,
        # Legacy parameters for backward compatibility
        kernel_size: int = 7,
        tophat_break: float = 100,
        min_distance: int = 2,
        local_max_thre: float = 200,
        intensity_thre: Optional[float] = None,
    ):
        # Setup background removal method
        if background_removal:
            bg_method = background_removal.get("method", "tophat")
            if bg_method == "tophat":
                kernel_size = background_removal.get("kernel_size", kernel_size)
                tophat_break = background_removal.get("tophat_break", tophat_break)
                self.background_remover = TophatBackgroundRemover(
                    kernel_size, tophat_break
                )
            else:
                raise ValueError(f"Unsupported background removal method: {bg_method}")
        else:
            # Use legacy parameters
            self.background_remover = TophatBackgroundRemover(kernel_size, tophat_break)

        # Setup coordinate detection method
        if coordinate_detection:
            coord_method = coordinate_detection.get("method", "local_max")
            if coord_method == "local_max":
                min_distance = coordinate_detection.get("min_distance", min_distance)
                local_max_thre = coordinate_detection.get(
                    "local_max_thre", local_max_thre
                )
                intensity_thre = coordinate_detection.get(
                    "intensity_thre", intensity_thre
                )
                self.coordinate_detector = LocalMaximaDetector(
                    min_distance, local_max_thre, intensity_thre
                )
            else:
                raise ValueError(
                    f"Unsupported coordinate detection method: {coord_method}"
                )
        else:
            # Use legacy parameters
            self.coordinate_detector = LocalMaximaDetector(
                min_distance, local_max_thre, intensity_thre
            )

        # Setup other components
        self.intensity_extractor = DirectIntensityExtractor()
        self.snr_calculator = SNRCalculator()
        self.duplicate_remover = DuplicateRemover(distance_threshold)

        self.snr_threshold = snr_threshold
        self.check_snr = check_snr
        self.remove_duplicates = remove_duplicates

    def detect_spots(
        self,
        image: np.ndarray,
        tophat_mean: Optional[float] = None,
        num_channels: int = 1,
    ) -> SpotDetectionResult:
        """
        检测信号点

        Args:
            image: 输入图像
            tophat_mean: Tophat图像均值（用于动态阈值）
            num_channels: 通道数量，用于计算内存阈值

        Returns:
            检测结果
        """
        # 1. 背景去除
        background_removed = self.background_remover.remove_background(image)

        # 2. 动态调整阈值
        if tophat_mean is None:
            tophat_mean = np.mean(background_removed)

        # 更新检测器阈值
        self.coordinate_detector.local_max_thre = max(
            self.coordinate_detector.local_max_thre, tophat_mean * self.snr_threshold
        )

        # 3. 坐标检测
        coordinates = self.coordinate_detector.detect_coordinates(background_removed)

        # 4. 计算SNR（可选）
        snr_values = None
        if self.check_snr:
            snr_values = self.snr_calculator.calculate_snr(image, coordinates)

        # 5. 移除重复点
        if self.remove_duplicates:
            coordinates = self.duplicate_remover.remove_duplicates(coordinates)

        # 6. 强度提取
        intensities = self.intensity_extractor.extract_intensities(
            background_removed, coordinates
        )

        # 7. 添加SNR信息
        if snr_values is not None:
            intensities["snr"] = snr_values

        # 8. 构建结果
        metadata = {
            "image_shape": image.shape,
            "num_spots": len(coordinates),
            "tophat_mean": tophat_mean,
            "snr_threshold": self.snr_threshold,
            "check_snr": self.check_snr,
            "remove_duplicates": self.remove_duplicates,
        }

        processing_info = {
            "method": "traditional",
            "kernel_size": self.background_remover.kernel_size,
            "tophat_break": self.background_remover.tophat_break,
        }

        return SpotDetectionResult(
            coordinates=coordinates,
            intensities=intensities,
            metadata=metadata,
            processing_info=processing_info,
        )


def create_traditional_detector(**kwargs) -> TraditionalSpotDetector:
    """创建传统检测器的工厂函数"""
    return TraditionalSpotDetector(**kwargs)


# 多通道处理支持
class MultiChannelTraditionalDetector:
    """多通道传统检测器"""

    def __init__(self, **detector_kwargs):
        self.detector = TraditionalSpotDetector(**detector_kwargs)
        self.image_processor = ImageProcessor()

    def detect_spots_multi_channel(
        self,
        image_dict: Dict[str, np.ndarray],
        merge_coordinates: bool = True,
        num_channels: int = None,
    ) -> SpotDetectionResult:
        """
        多通道信号点检测

        Args:
            image_dict: 通道名称到图像的映射
            merge_coordinates: 是否合并各通道的坐标

        Returns:
            检测结果
        """
        all_coordinates = []
        all_intensities = {}
        channel_metadata = {}

        # 处理每个通道
        for channel, image in image_dict.items():
            logger.info(f"Processing channel: {channel}")

            # 检测该通道的信号点
            result = self.detector.detect_spots(
                image, num_channels=num_channels or len(image_dict)
            )

            if merge_coordinates:
                all_coordinates.append(result.coordinates)

            # 收集强度信息
            for intensity_key, intensity_values in result.intensities.items():
                key = f"{channel}_{intensity_key}"
                all_intensities[key] = intensity_values

            channel_metadata[channel] = result.metadata

        # 合并坐标
        if merge_coordinates and all_coordinates:
            # 使用重复点移除器合并坐标
            merged_coordinates = np.vstack(all_coordinates)
            merged_coordinates = self.detector.duplicate_remover.remove_duplicates(
                merged_coordinates
            )
        else:
            merged_coordinates = (
                np.vstack(all_coordinates) if all_coordinates else np.empty((0, 2))
            )

        # 重新提取所有通道的强度
        if merge_coordinates and len(merged_coordinates) > 0:
            for channel, image in image_dict.items():
                # 获取背景去除后的图像
                background_removed = self.image_processor.get_processed_image(
                    f"{channel}_bg_removed",
                    lambda: self.detector.background_remover.remove_background(image),
                )

                # 提取强度
                intensities = self.detector.intensity_extractor.extract_intensities(
                    background_removed, merged_coordinates
                )

                for intensity_key, intensity_values in intensities.items():
                    key = f"{channel}_{intensity_key}"
                    all_intensities[key] = intensity_values

        # 构建最终结果
        metadata = {
            "channels": list(image_dict.keys()),
            "num_spots": len(merged_coordinates),
            "merge_coordinates": merge_coordinates,
            "channel_metadata": channel_metadata,
        }

        processing_info = {
            "method": "multi_channel_traditional",
            "num_channels": len(image_dict),
        }

        return SpotDetectionResult(
            coordinates=merged_coordinates,
            intensities=all_intensities,
            metadata=metadata,
            processing_info=processing_info,
        )
