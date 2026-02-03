"""
统一的信号点检测器
Unified spot detector

整合所有检测方法和强度提取策略的统一接口
Unified interface integrating all detection methods and intensity extraction strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import yaml

from .spot_detection import SpotDetectionPipeline, SpotDetectionResult, ImageProcessor
from .traditional_methods import (
    TraditionalSpotDetector,
    MultiChannelTraditionalDetector,
)
from .deep_learning_methods import (
    DeepLearningSpotDetector,
    MultiChannelDeepLearningDetector,
    create_deep_learning_detector,
    create_multi_channel_deep_learning_detector,
)
from .intensity_extractors import (
    create_intensity_extractor,
    create_preset_intensity_extractor,
    INTENSITY_EXTRACTOR_CONFIGS,
)

logger = logging.getLogger(__name__)


class UnifiedSpotDetector:
    """统一的信号点检测器"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化统一检测器

        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        self.image_processor = ImageProcessor(
            max_memory_gb=self.config.get("max_memory", 8.0)
        )
        self._detector = None
        self._intensity_extractor = None

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "method": "traditional",  # 'traditional', 'deep_learning'
            "max_memory_gb": 8.0,
            "use_tiling": True,
            "traditional": {
                "kernel_size": 7,
                "tophat_break": 100,
                "min_distance": 2,
                "local_max_thre": 200,
                "intensity_thre": None,
                "snr_threshold": 8.0,
                "check_snr": False,
                "remove_duplicates": True,
                "distance_threshold": 2.0,
            },
            "deep_learning": {
                "model_path": "",
                "model_name": "",
                "prob_thresh": 0.5,
                "nms_thresh": 0.3,
                "roi_size": 15,
                "intensity_method": "gaussian",
            },
            "intensity_extraction": {
                "method": "gaussian",  # 'direct', 'gaussian', 'mask', 'integrated', 'adaptive', 'multiscale'
                "preset": None,  # 如果指定preset，会覆盖method
                "roi_size": 15,
                "mask_radius": 3,
                "background_radius": 8,
                "background_method": "mean",
            },
        }

    def setup_detector(self, method: Optional[str] = None, **kwargs):
        """设置检测器"""
        method = method or self.config["method"]

        if method == "traditional":
            traditional_config = self.config.get("traditional", {})
            # Pass the configuration directly to TraditionalSpotDetector
            self._detector = TraditionalSpotDetector(**{**traditional_config, **kwargs})
        elif method == "deep_learning":
            deep_learning_config = self.config.get("deep_learning", {})
            if not deep_learning_config.get("model_path"):
                raise ValueError("Model path is required for deep learning method")

            self._detector = DeepLearningSpotDetector(
                **{**deep_learning_config, **kwargs}
            )
        else:
            raise ValueError(f"Unknown detection method: {method}")

        logger.info(f"Detector setup with method: {method}")

    def setup_intensity_extractor(self, method: Optional[str] = None, **kwargs):
        """设置强度提取器"""
        intensity_config = self.config["intensity_extraction"]

        # 如果指定了preset，使用preset
        if intensity_config.get("preset"):
            self._intensity_extractor = create_preset_intensity_extractor(
                intensity_config["preset"]
            )
        else:
            # 否则使用method
            method = method or intensity_config["method"]
            # 只传递对应方法的参数
            method_params = intensity_config.get(method, {})
            self._intensity_extractor = create_intensity_extractor(
                method, **{**method_params, **kwargs}
            )

        logger.info(f"Intensity extractor setup with method: {method}")

    def detect_spots(
        self,
        image: Union[np.ndarray, Dict[str, np.ndarray]],
        method: Optional[str] = None,
        intensity_method: Optional[str] = None,
        **kwargs,
    ) -> SpotDetectionResult:
        """
        检测信号点

        Args:
            image: 输入图像（单通道或多通道字典）
            method: 检测方法
            intensity_method: 强度提取方法
            **kwargs: 其他参数

        Returns:
            检测结果
        """
        # 设置检测器
        if method or not self._detector:
            self.setup_detector(method, **kwargs)

        # 设置强度提取器
        if intensity_method or not self._intensity_extractor:
            self.setup_intensity_extractor(intensity_method, **kwargs)

        # 处理单通道图像
        if isinstance(image, np.ndarray):
            return self._detect_single_channel(image)

        # 处理多通道图像
        elif isinstance(image, dict):
            return self._detect_multi_channel(image)

        else:
            raise ValueError("Image must be numpy array or dictionary of channels")

    def _detect_single_channel(self, image: np.ndarray) -> SpotDetectionResult:
        """单通道检测"""
        # 检测坐标
        if isinstance(self._detector, TraditionalSpotDetector):
            result = self._detector.detect_spots(image, num_channels=1)
        elif isinstance(self._detector, DeepLearningSpotDetector):
            result = self._detector.detect_spots(image, num_channels=1)
        else:
            raise ValueError("Unknown detector type")

        # 重新提取强度（使用指定的强度提取器）
        if self._intensity_extractor:
            intensities = self._intensity_extractor.extract_intensities(
                image, result.coordinates
            )
            result.intensities = intensities

        return result

    def _detect_multi_channel(
        self, image_dict: Dict[str, np.ndarray]
    ) -> SpotDetectionResult:
        """多通道检测"""
        num_channels = len(image_dict)

        if isinstance(self._detector, TraditionalSpotDetector):
            # 使用多通道传统检测器
            multi_detector = MultiChannelTraditionalDetector(
                **self.config["traditional"]
            )
            result = multi_detector.detect_spots_multi_channel(
                image_dict, num_channels=num_channels
            )

        elif isinstance(self._detector, DeepLearningSpotDetector):
            # 使用多通道深度学习检测器
            multi_detector = MultiChannelDeepLearningDetector(
                **self.config["deep_learning"]
            )
            result = multi_detector.detect_spots_multi_channel(
                image_dict, num_channels=num_channels
            )

        else:
            raise ValueError("Unknown detector type")

        # 重新提取强度（使用指定的强度提取器）
        if self._intensity_extractor:
            all_intensities = {}
            for channel, image in image_dict.items():
                intensities = self._intensity_extractor.extract_intensities(
                    image, result.coordinates
                )
                for intensity_key, intensity_values in intensities.items():
                    key = f"{channel}_{intensity_key}"
                    all_intensities[key] = intensity_values

            result.intensities = all_intensities

        return result

    def save_results(self, result: SpotDetectionResult, output_path: str):
        """保存检测结果"""
        output_path = Path(output_path)

        # 保存坐标
        coords_df = pd.DataFrame(
            {"Y": result.coordinates[:, 0], "X": result.coordinates[:, 1]}
        )
        coords_df.to_csv(output_path.with_suffix(".coordinates.csv"), index=False)

        # 保存强度
        if result.intensities is not None:
            if isinstance(result.intensities, pd.DataFrame):
                intensity_df = result.intensities
            else:
                intensity_df = pd.DataFrame(result.intensities)
            intensity_df.to_csv(
                output_path.with_suffix(".intensities.csv"), index=False
            )

        # 保存元数据
        metadata_df = pd.DataFrame([result.metadata])
        metadata_df.to_csv(output_path.with_suffix(".metadata.csv"), index=False)

        logger.info(f"Results saved to {output_path}")

    def load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Config loaded from {config_path}")

    def save_config(self, config_path: str):
        """保存配置文件"""
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Config saved to {config_path}")


# 便捷函数
def detect_spots(
    image: Union[np.ndarray, Dict[str, np.ndarray]],
    method: str = "traditional",
    intensity_method: str = "gaussian",
    **kwargs,
) -> SpotDetectionResult:
    """
    便捷的信号点检测函数

    Args:
        image: 输入图像
        method: 检测方法 ('traditional', 'deep_learning')
        intensity_method: 强度提取方法
        **kwargs: 其他参数

    Returns:
        检测结果
    """
    detector = UnifiedSpotDetector()
    return detector.detect_spots(
        image, method=method, intensity_method=intensity_method, **kwargs
    )


def create_detector_from_config(config_path: str) -> UnifiedSpotDetector:
    """从配置文件创建检测器"""
    detector = UnifiedSpotDetector()
    detector.load_config(config_path)
    return detector


# 预定义配置
DEFAULT_CONFIGS = {
    "traditional_fast": {
        "method": "traditional",
        "traditional": {
            "kernel_size": 5,
            "tophat_break": 50,
            "min_distance": 1,
            "local_max_thre": 100,
            "remove_duplicates": False,
        },
        "intensity_extraction": {
            "method": "direct",
            "kernel_size": 3,
            "dilation_iterations": 0,
        },
    },
    "traditional_accurate": {
        "method": "traditional",
        "traditional": {
            "kernel_size": 7,
            "tophat_break": 100,
            "min_distance": 2,
            "local_max_thre": 200,
            "snr_threshold": 8.0,
            "check_snr": True,
            "remove_duplicates": True,
        },
        "intensity_extraction": {"method": "gaussian", "roi_size": 15},
    },
    "deep_learning_standard": {
        "method": "deep_learning",
        "deep_learning": {
            "prob_thresh": 0.5,
            "nms_thresh": 0.3,
            "roi_size": 15,
            "intensity_method": "gaussian",
        },
        "intensity_extraction": {"method": "gaussian", "roi_size": 15},
    },
}


def create_preset_detector(preset: str, **kwargs) -> UnifiedSpotDetector:
    """使用预定义配置创建检测器"""
    if preset not in DEFAULT_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset}. Available presets: {list(DEFAULT_CONFIGS.keys())}"
        )

    config = DEFAULT_CONFIGS[preset].copy()
    config.update(kwargs)

    detector = UnifiedSpotDetector(config)
    return detector


if __name__ == "__main__":
    # 示例使用
    import numpy as np

    # 创建测试图像
    test_image = np.random.rand(1000, 1000) * 1000

    # 使用便捷函数
    result = detect_spots(test_image, method="traditional", intensity_method="gaussian")
    print(f"检测到 {len(result.coordinates)} 个信号点")

    # 使用预定义配置
    detector = create_preset_detector("traditional_accurate")
    result = detector.detect_spots(test_image)
    print(f"使用预定义配置检测到 {len(result.coordinates)} 个信号点")
