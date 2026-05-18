"""
统一的强度提取模块
Unified intensity extraction module

整合所有强度提取方法，支持多种策略
Integrates all intensity extraction methods with multiple strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm
import cv2

from .spot_detection import IntensityExtractor

logger = logging.getLogger(__name__)


class DirectIntensityExtractor(IntensityExtractor):
    """直接强度提取器 - 直接读取坐标位置的像素值"""

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


class GaussianIntensityExtractor(IntensityExtractor):
    """高斯拟合强度提取器"""

    def __init__(
        self,
        roi_size: int = 15,
        fit_failed_fallback: bool = True,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        self.roi_size = roi_size
        self.fit_failed_fallback = fit_failed_fallback
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def extract_intensities(
        self, image: np.ndarray, coordinates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """使用高斯拟合提取强度"""
        if len(coordinates) == 0:
            return {
                "intensity": np.array([]),
                "background": np.array([]),
                "amplitude": np.array([]),
                "sigma_x": np.array([]),
                "sigma_y": np.array([]),
                "fit_success": np.array([]),
            }

        intensities = []
        backgrounds = []
        amplitudes = []
        sigma_xs = []
        sigma_ys = []
        fit_success = []

        # Import gaussian fitting functions when needed
        from .gaussian_fitting import fit_gaussian_2d, get_intensity_and_background

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
                fit_success.append(True)
            else:
                # 拟合失败
                fit_success.append(False)
                if self.fit_failed_fallback:
                    # 使用直接读取作为备选
                    intensities.append(image[y, x])
                    backgrounds.append(0)
                    amplitudes.append(image[y, x])
                    sigma_xs.append(1.0)
                    sigma_ys.append(1.0)
                else:
                    intensities.append(0)
                    backgrounds.append(0)
                    amplitudes.append(0)
                    sigma_xs.append(0)
                    sigma_ys.append(0)

        return {
            "intensity": np.array(intensities),
            "background": np.array(backgrounds),
            "amplitude": np.array(amplitudes),
            "sigma_x": np.array(sigma_xs),
            "sigma_y": np.array(sigma_ys),
            "fit_success": np.array(fit_success),
        }


class MaskIntensityExtractor(IntensityExtractor):
    """掩码区域强度提取器"""

    def __init__(
        self,
        mask_radius: int = 3,
        background_radius: int = 8,
        background_method: str = "mean",
    ):
        self.mask_radius = mask_radius
        self.background_radius = background_radius
        self.background_method = background_method  # 'mean', 'median', 'min'

    def extract_intensities(
        self, image: np.ndarray, coordinates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """使用掩码区域提取强度"""
        if len(coordinates) == 0:
            return {
                "intensity": np.array([]),
                "background": np.array([]),
                "snr": np.array([]),
                "signal_area": np.array([]),
            }

        intensities = []
        backgrounds = []
        snrs = []
        signal_areas = []

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
            signal_area = np.sum(signal_mask)

            if np.any(background_mask):
                if self.background_method == "mean":
                    background_intensity = np.mean(image[background_mask])
                elif self.background_method == "median":
                    background_intensity = np.median(image[background_mask])
                elif self.background_method == "min":
                    background_intensity = np.min(image[background_mask])
                else:
                    background_intensity = np.mean(image[background_mask])
            else:
                background_intensity = 0

            snr = (
                signal_intensity / background_intensity
                if background_intensity > 0
                else float("inf")
            )

            intensities.append(signal_intensity)
            backgrounds.append(background_intensity)
            snrs.append(snr)
            signal_areas.append(signal_area)

        return {
            "intensity": np.array(intensities),
            "background": np.array(backgrounds),
            "snr": np.array(snrs),
            "signal_area": np.array(signal_areas),
        }


class IntegratedIntensityExtractor(IntensityExtractor):
    """积分强度提取器 - 计算高斯积分的理论值"""

    def __init__(self, roi_size: int = 15, integration_method: str = "gaussian"):
        self.roi_size = roi_size
        self.integration_method = integration_method

    def extract_intensities(
        self, image: np.ndarray, coordinates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """使用积分方法提取强度"""
        if len(coordinates) == 0:
            return {
                "integrated_intensity": np.array([]),
                "background": np.array([]),
                "amplitude": np.array([]),
                "sigma_x": np.array([]),
                "sigma_y": np.array([]),
            }

        integrated_intensities = []
        backgrounds = []
        amplitudes = []
        sigma_xs = []
        sigma_ys = []

        # Import gaussian fitting functions when needed
        from .gaussian_fitting import fit_gaussian_2d, get_intensity_and_background

        for y, x in tqdm(coordinates, desc="Integrated intensity", leave=False):
            popt, _ = fit_gaussian_2d(image, y, x, roi_size=self.roi_size)

            if popt is not None:
                amplitude, _, _, sigma_x, sigma_y, _, offset = popt

                # 计算高斯积分：2 * pi * A * sigma_x * sigma_y
                integrated_intensity = 2 * np.pi * amplitude * sigma_x * sigma_y

                integrated_intensities.append(integrated_intensity)
                backgrounds.append(offset)
                amplitudes.append(amplitude)
                sigma_xs.append(sigma_x)
                sigma_ys.append(sigma_y)
            else:
                # 拟合失败，使用直接读取
                integrated_intensities.append(image[y, x])
                backgrounds.append(0)
                amplitudes.append(image[y, x])
                sigma_xs.append(1.0)
                sigma_ys.append(1.0)

        return {
            "integrated_intensity": np.array(integrated_intensities),
            "background": np.array(backgrounds),
            "amplitude": np.array(amplitudes),
            "sigma_x": np.array(sigma_xs),
            "sigma_y": np.array(sigma_ys),
        }


class AdaptiveIntensityExtractor(IntensityExtractor):
    """自适应强度提取器 - 根据信号特征选择最佳方法"""

    def __init__(
        self,
        gaussian_roi_size: int = 15,
        mask_radius: int = 3,
        background_radius: int = 8,
        snr_threshold: float = 3.0,
    ):
        self.gaussian_extractor = GaussianIntensityExtractor(gaussian_roi_size)
        self.mask_extractor = MaskIntensityExtractor(mask_radius, background_radius)
        self.direct_extractor = DirectIntensityExtractor()
        self.snr_threshold = snr_threshold

    def extract_intensities(
        self, image: np.ndarray, coordinates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """自适应选择强度提取方法"""
        if len(coordinates) == 0:
            return {"intensity": np.array([])}

        # 首先使用掩码方法快速评估SNR
        mask_result = self.mask_extractor.extract_intensities(image, coordinates)
        snrs = mask_result["snr"]

        # 根据SNR选择方法
        high_snr_mask = snrs > self.snr_threshold
        low_snr_mask = ~high_snr_mask

        # 高SNR使用高斯拟合
        if np.any(high_snr_mask):
            high_snr_coords = coordinates[high_snr_mask]
            gaussian_result = self.gaussian_extractor.extract_intensities(
                image, high_snr_coords
            )
        else:
            gaussian_result = {"intensity": np.array([])}

        # 低SNR使用直接读取
        if np.any(low_snr_mask):
            low_snr_coords = coordinates[low_snr_mask]
            direct_result = self.direct_extractor.extract_intensities(
                image, low_snr_coords
            )
        else:
            direct_result = {"intensity": np.array([])}

        # 合并结果
        final_intensities = np.zeros(len(coordinates))
        if len(gaussian_result["intensity"]) > 0:
            final_intensities[high_snr_mask] = gaussian_result["intensity"]
        if len(direct_result["intensity"]) > 0:
            final_intensities[low_snr_mask] = direct_result["intensity"]

        return {
            "intensity": final_intensities,
            "method_used": np.where(high_snr_mask, "gaussian", "direct"),
            "snr": snrs,
        }


class MultiScaleIntensityExtractor(IntensityExtractor):
    """多尺度强度提取器 - 在不同尺度上提取强度"""

    def __init__(self, scales: List[int] = [1, 2, 3, 5]):
        self.scales = scales
        self.mask_extractors = {
            scale: MaskIntensityExtractor(
                mask_radius=scale, background_radius=scale * 2
            )
            for scale in scales
        }

    def extract_intensities(
        self, image: np.ndarray, coordinates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """在多尺度上提取强度"""
        if len(coordinates) == 0:
            return {f"intensity_scale_{scale}": np.array([]) for scale in self.scales}

        results = {}

        for scale, extractor in self.mask_extractors.items():
            result = extractor.extract_intensities(image, coordinates)
            results[f"intensity_scale_{scale}"] = result["intensity"]
            results[f"background_scale_{scale}"] = result["background"]
            results[f"snr_scale_{scale}"] = result["snr"]

        return results


# 工厂函数
def create_intensity_extractor(method: str, **kwargs) -> IntensityExtractor:
    """创建强度提取器的工厂函数"""
    if method == "direct":
        return DirectIntensityExtractor(**kwargs)
    elif method == "gaussian":
        return GaussianIntensityExtractor(**kwargs)
    elif method == "mask":
        return MaskIntensityExtractor(**kwargs)
    elif method == "integrated":
        return IntegratedIntensityExtractor(**kwargs)
    elif method == "adaptive":
        return AdaptiveIntensityExtractor(**kwargs)
    elif method == "multiscale":
        return MultiScaleIntensityExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown intensity extraction method: {method}")


# 预定义的强度提取器配置
INTENSITY_EXTRACTOR_CONFIGS = {
    "fast": {"method": "direct", "kernel_size": 3, "dilation_iterations": 0},
    "accurate": {"method": "gaussian", "roi_size": 15, "fit_failed_fallback": True},
    "robust": {
        "method": "mask",
        "mask_radius": 3,
        "background_radius": 8,
        "background_method": "median",
    },
    "theoretical": {"method": "integrated", "roi_size": 15},
    "adaptive": {
        "method": "adaptive",
        "gaussian_roi_size": 15,
        "mask_radius": 3,
        "background_radius": 8,
        "snr_threshold": 3.0,
    },
    "multiscale": {"method": "multiscale", "scales": [1, 2, 3, 5]},
}


def create_preset_intensity_extractor(preset: str) -> IntensityExtractor:
    """使用预定义配置创建强度提取器"""
    if preset not in INTENSITY_EXTRACTOR_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset}. Available presets: {list(INTENSITY_EXTRACTOR_CONFIGS.keys())}"
        )

    config = INTENSITY_EXTRACTOR_CONFIGS[preset]
    method = config.pop("method")
    return create_intensity_extractor(method, **config)
