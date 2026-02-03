"""
Maintainable spot detection framework

Supports multiple detection methods and intensity extraction strategies with high extensibility
"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from tqdm import tqdm
import cv2

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpotDetectionResult:
    """信号点检测结果数据类"""

    coordinates: np.ndarray  # 坐标 (N, 2) - (y, x)
    intensities: Dict[str, np.ndarray]  # 各通道强度 {channel: intensity_array}
    metadata: Dict[str, Any]  # 元数据信息
    processing_info: Dict[str, Any]  # 处理信息


class ImageProcessor:
    """图像处理管理器 - 处理图像缓存、分块和内存管理"""

    def __init__(self, max_memory_gb: float = 8.0):
        self.image_cache: Dict[str, np.ndarray] = {}
        self.max_memory_gb = max_memory_gb
        self.tile_processor = TileProcessor()

    def get_processed_image(
        self, image_key: str, processor_func, force_recompute: bool = False
    ) -> np.ndarray:
        """
        获取处理过的图像，支持缓存

        Args:
            image_key: 图像缓存键
            processor_func: 处理函数
            force_recompute: 是否强制重新计算

        Returns:
            处理后的图像
        """
        if force_recompute or image_key not in self.image_cache:
            logger.info(f"Processing image: {image_key}")
            self.image_cache[image_key] = processor_func()
            self._check_memory_usage()

        return self.image_cache[image_key]

    def clear_cache(self):
        """清空图像缓存"""
        self.image_cache.clear()
        logger.info("Image cache cleared")

    def _check_memory_usage(self):
        """检查内存使用情况"""
        total_memory = sum(arr.nbytes for arr in self.image_cache.values()) / (1024**3)
        if total_memory > self.max_memory_gb:
            logger.warning(f"Memory usage high: {total_memory:.2f}GB")
            # 可以在这里实现LRU缓存清理策略


class TileProcessor:
    """分块处理器 - 处理特大图片的分块处理"""

    def __init__(self, max_volume: int = 10**8, overlap: int = 500):
        self.max_volume = max_volume
        self.overlap = overlap

    def process_large_image(
        self, image: np.ndarray, processor_func, **kwargs
    ) -> SpotDetectionResult:
        """
        分块处理大图像

        Args:
            image: 输入图像
            processor_func: 处理函数
            **kwargs: 其他参数

        Returns:
            合并后的检测结果
        """
        if len(image.shape) == 3:
            zrange, yrange, xrange = image.shape
        elif len(image.shape) == 2:
            yrange, xrange = image.shape
            zrange = 1
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # 计算分块参数
        xy_size = int(np.sqrt(self.max_volume / zrange))
        x_num = -(-(xrange - self.overlap) // (xy_size - self.overlap))
        y_num = -(-(yrange - self.overlap) // (xy_size - self.overlap))
        cut_x = xrange // x_num + self.overlap
        cut_y = yrange // y_num + self.overlap

        logger.info(f"Processing large image in {x_num * y_num} tiles")
        logger.info(f"Tile size: {cut_x} x {cut_y}, overlap: {self.overlap}")

        all_results = []

        with tqdm(total=x_num * y_num, desc="Processing tiles") as pbar:
            for x_pos in range(x_num):
                pad_x = x_pos * (cut_x - self.overlap)
                for y_pos in range(y_num):
                    pad_y = y_pos * (cut_y - self.overlap)

                    # 提取图像块
                    if len(image.shape) == 3:
                        tile = image[:, pad_y : pad_y + cut_y, pad_x : pad_x + cut_x]
                    else:
                        tile = image[pad_y : pad_y + cut_y, pad_x : pad_x + cut_x]

                    # 处理图像块
                    tile_result = processor_func(
                        tile, pad_x=pad_x, pad_y=pad_y, **kwargs
                    )

                    if tile_result is not None:
                        all_results.append(tile_result)

                    pbar.update(1)

        # 合并结果
        return self._merge_results(all_results)

    def _merge_results(self, results: List[SpotDetectionResult]) -> SpotDetectionResult:
        """合并分块处理结果"""
        if not results:
            return SpotDetectionResult(
                coordinates=np.empty((0, 2)),
                intensities={},
                metadata={},
                processing_info={},
            )

        # 合并坐标
        all_coords = np.vstack([r.coordinates for r in results])

        # 合并强度
        all_intensities = {}
        for channel in results[0].intensities.keys():
            all_intensities[channel] = np.concatenate(
                [r.intensities[channel] for r in results]
            )

        # 合并元数据
        merged_metadata = {}
        for r in results:
            for key, value in r.metadata.items():
                if key not in merged_metadata:
                    merged_metadata[key] = []
                merged_metadata[key].append(value)

        # 合并处理信息
        merged_processing_info = {
            "num_tiles": len(results),
            "total_spots": len(all_coords),
        }

        return SpotDetectionResult(
            coordinates=all_coords,
            intensities=all_intensities,
            metadata=merged_metadata,
            processing_info=merged_processing_info,
        )

    def process_large_image_with_memmap(
        self, memmap_loader, processor_func, image_shape: tuple, **kwargs
    ) -> SpotDetectionResult:
        """
        使用内存映射分块处理大图像

        Args:
            memmap_loader: 内存映射加载器
            processor_func: 处理函数
            image_shape: 图像形状
            **kwargs: 其他参数

        Returns:
            合并后的检测结果
        """
        if len(image_shape) == 3:
            zrange, yrange, xrange = image_shape
        elif len(image_shape) == 2:
            yrange, xrange = image_shape
            zrange = 1
        else:
            raise ValueError(f"Unsupported image shape: {image_shape}")

        # 计算分块参数
        xy_size = int(np.sqrt(self.max_volume / zrange))
        x_num = -(-(xrange - self.overlap) // (xy_size - self.overlap))
        y_num = -(-(yrange - self.overlap) // (xy_size - self.overlap))
        cut_x = xrange // x_num + self.overlap
        cut_y = yrange // y_num + self.overlap

        logger.info(f"Processing large image with memmap in {x_num * y_num} tiles")
        logger.info(f"Tile size: {cut_x} x {cut_y}, overlap: {self.overlap}")

        all_results = []

        with tqdm(total=x_num * y_num, desc="Processing tiles with memmap") as pbar:
            for x_pos in range(x_num):
                pad_x = x_pos * (cut_x - self.overlap)
                for y_pos in range(y_num):
                    pad_y = y_pos * (cut_y - self.overlap)

                    # 使用内存映射加载图像块
                    tile_dict = memmap_loader.load_all_channel_tiles(
                        pad_x, pad_y, cut_x, cut_y
                    )

                    # 处理图像块
                    tile_result = processor_func(
                        tile_dict, pad_x=pad_x, pad_y=pad_y, **kwargs
                    )

                    if tile_result is not None:
                        all_results.append(tile_result)

                    pbar.update(1)

        # 合并结果
        return self._merge_results(all_results)


class BackgroundRemover(ABC):
    """背景去除抽象基类"""

    @abstractmethod
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """去除背景"""
        pass


class TophatBackgroundRemover(BackgroundRemover):
    """Tophat背景去除器"""

    def __init__(self, kernel_size: int = 7):
        self.kernel_size = kernel_size

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """使用Tophat形态学操作去除背景"""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


class CoordinateDetector(ABC):
    """坐标检测抽象基类"""

    @abstractmethod
    def detect_coordinates(self, image: np.ndarray) -> np.ndarray:
        """检测信号点坐标"""
        pass


class LocalMaximaDetector(CoordinateDetector):
    """局部最大值检测器"""

    def __init__(self, min_distance: int = 2, threshold_abs: float = 200):
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs

    def detect_coordinates(self, image: np.ndarray) -> np.ndarray:
        """检测局部最大值坐标"""
        from skimage.feature import peak_local_maxima

        coordinates = peak_local_maxima(
            image, min_distance=self.min_distance, threshold_abs=self.threshold_abs
        )
        return np.array(coordinates)


class IntensityExtractor(ABC):
    """强度提取抽象基类"""

    @abstractmethod
    def extract_intensities(
        self, image: np.ndarray, coordinates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """提取强度值"""
        pass


class DirectIntensityExtractor(IntensityExtractor):
    """直接强度提取器 - 直接读取坐标位置的像素值"""

    def extract_intensities(
        self, image: np.ndarray, coordinates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """直接读取坐标位置的强度值"""
        if len(coordinates) == 0:
            return {"intensity": np.array([])}

        intensities = image[coordinates[:, 0], coordinates[:, 1]]
        return {"intensity": intensities}


class SpotDetectionPipeline:
    """信号点检测主流程控制器"""

    def __init__(
        self,
        background_remover: BackgroundRemover,
        coordinate_detector: CoordinateDetector,
        intensity_extractor: IntensityExtractor,
        image_processor: Optional[ImageProcessor] = None,
    ):
        self.background_remover = background_remover
        self.coordinate_detector = coordinate_detector
        self.intensity_extractor = intensity_extractor
        self.image_processor = image_processor or ImageProcessor()

    def detect_spots(
        self,
        image: np.ndarray,
        use_tiling: bool = True,
        num_channels: int = 1,
        **kwargs,
    ) -> SpotDetectionResult:
        """
        检测信号点

        Args:
            image: 输入图像
            use_tiling: 是否使用分块处理
            num_channels: 通道数量，用于计算内存阈值
            **kwargs: 其他参数

        Returns:
            检测结果
        """
        # 根据最大内存和通道数动态计算阈值
        # 假设每个像素占用4字节(float32)，乘以通道数*2(原始+处理后)
        max_memory_bytes = self.image_processor.max_memory_gb * 1024**3
        bytes_per_pixel = 4  # float32
        memory_factor = num_channels * 2  # 原始图像 + 处理后图像
        max_pixels = max_memory_bytes // (bytes_per_pixel * memory_factor)

        logger.info(
            f"Memory-based tiling threshold: {max_pixels:,} pixels "
            f"(max_memory: {self.image_processor.max_memory_gb}GB, "
            f"channels: {num_channels}, factor: {memory_factor})"
        )
        logger.info(
            f"Image size: {image.size:,} pixels, tiling needed: {image.size > max_pixels}"
        )

        if use_tiling and image.size > max_pixels:  # 根据内存动态判断是否分块
            return self.image_processor.tile_processor.process_large_image(
                image, self._process_single_tile, **kwargs
            )
        else:
            return self._process_single_tile(image, **kwargs)

    def _process_single_tile(
        self, image: np.ndarray, pad_x: int = 0, pad_y: int = 0, **kwargs
    ) -> SpotDetectionResult:
        """处理单个图像块"""

        # 1. 背景去除
        background_removed = self.image_processor.get_processed_image(
            f"bg_removed_{id(image)}",
            lambda: self.background_remover.remove_background(image),
        )

        # 2. 坐标检测
        coordinates = self.coordinate_detector.detect_coordinates(background_removed)

        # 3. 强度提取
        intensities = self.intensity_extractor.extract_intensities(
            background_removed, coordinates
        )

        # 4. 构建结果
        metadata = {
            "image_shape": image.shape,
            "num_spots": len(coordinates),
            "background_remover": type(self.background_remover).__name__,
            "coordinate_detector": type(self.coordinate_detector).__name__,
            "intensity_extractor": type(self.intensity_extractor).__name__,
        }

        processing_info = {"pad_x": pad_x, "pad_y": pad_y, "tile_processed": True}

        return SpotDetectionResult(
            coordinates=coordinates,
            intensities=intensities,
            metadata=metadata,
            processing_info=processing_info,
        )


# 工厂函数
def create_traditional_pipeline(
    kernel_size: int = 7, min_distance: int = 2, threshold_abs: float = 200
) -> SpotDetectionPipeline:
    """创建传统方法检测流程"""
    background_remover = TophatBackgroundRemover(kernel_size)
    coordinate_detector = LocalMaximaDetector(min_distance, threshold_abs)
    intensity_extractor = DirectIntensityExtractor()

    return SpotDetectionPipeline(
        background_remover=background_remover,
        coordinate_detector=coordinate_detector,
        intensity_extractor=intensity_extractor,
    )


if __name__ == "__main__":
    # 示例使用
    pipeline = create_traditional_pipeline()

    # 假设有一个测试图像
    test_image = np.random.rand(1000, 1000) * 1000

    result = pipeline.detect_spots(test_image)
    print(f"检测到 {len(result.coordinates)} 个信号点")
    print(
        f"强度范围: {result.intensities['intensity'].min():.2f} - {result.intensities['intensity'].max():.2f}"
    )
