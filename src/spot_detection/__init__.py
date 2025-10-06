"""
Spot detection module for multi-channel image processing
Maintainable spot detection framework with lazy imports for optional dependencies
"""

# Core classes that don't require heavy dependencies
from .spot_detection import SpotDetectionResult, ImageProcessor, TileProcessor
from .memmap_loader import (
    MemmapImageLoader,
    MultiChannelMemmapLoader,
    estimate_memory_usage,
)

# Configuration and channel management (no external dependencies)
from .config_loader import (
    load_config,
    load_batch_config,
    merge_configs,
    load_spot_detection_config,
    create_config_from_modules,
)

from .channel_manager import (
    ChannelManager,
    create_channel_manager_from_config,
)


# Delayed imports for optional dependencies
def _import_traditional_methods():
    """Delayed import of traditional methods (requires skimage)"""
    from .traditional_methods import (
        TophatBackgroundRemover,
        LocalMaximaDetector,
        DuplicateRemover,
        SNRCalculator,
        TraditionalSpotDetector,
        MultiChannelTraditionalDetector,
    )

    return {
        "TophatBackgroundRemover": TophatBackgroundRemover,
        "LocalMaximaDetector": LocalMaximaDetector,
        "DuplicateRemover": DuplicateRemover,
        "SNRCalculator": SNRCalculator,
        "TraditionalSpotDetector": TraditionalSpotDetector,
        "MultiChannelTraditionalDetector": MultiChannelTraditionalDetector,
    }


def _import_intensity_extractors():
    """Delayed import of intensity extractors (requires scipy)"""
    from .intensity_extractors import (
        DirectIntensityExtractor,
        GaussianIntensityExtractor,
        MaskIntensityExtractor,
        IntegratedIntensityExtractor,
        AdaptiveIntensityExtractor,
        MultiScaleIntensityExtractor,
        create_intensity_extractor,
        create_preset_intensity_extractor,
        INTENSITY_EXTRACTOR_CONFIGS,
    )

    return {
        "DirectIntensityExtractor": DirectIntensityExtractor,
        "GaussianIntensityExtractor": GaussianIntensityExtractor,
        "MaskIntensityExtractor": MaskIntensityExtractor,
        "IntegratedIntensityExtractor": IntegratedIntensityExtractor,
        "AdaptiveIntensityExtractor": AdaptiveIntensityExtractor,
        "MultiScaleIntensityExtractor": MultiScaleIntensityExtractor,
        "create_intensity_extractor": create_intensity_extractor,
        "create_preset_intensity_extractor": create_preset_intensity_extractor,
        "INTENSITY_EXTRACTOR_CONFIGS": INTENSITY_EXTRACTOR_CONFIGS,
    }


def _import_deep_learning_methods():
    """Delayed import of deep learning methods (requires stardist/tensorflow)"""
    from .deep_learning_methods import (
        StarDistDetector,
        DeepLearningSpotDetector,
        MultiChannelDeepLearningDetector,
    )

    return {
        "StarDistDetector": StarDistDetector,
        "DeepLearningSpotDetector": DeepLearningSpotDetector,
        "MultiChannelDeepLearningDetector": MultiChannelDeepLearningDetector,
    }


def _import_unified_detector():
    """Delayed import of unified detector (requires all methods)"""
    from .unified_detector import (
        UnifiedSpotDetector,
        detect_spots,
        create_detector_from_config,
        create_preset_detector,
        DEFAULT_CONFIGS,
    )

    return {
        "UnifiedSpotDetector": UnifiedSpotDetector,
        "detect_spots": detect_spots,
        "create_detector_from_config": create_detector_from_config,
        "create_preset_detector": create_preset_detector,
        "DEFAULT_CONFIGS": DEFAULT_CONFIGS,
    }


def _import_legacy_functions():
    """Delayed import of legacy processor functions (requires skimage)"""
    from .legacy.multi_channel_processor import (
        tophat_spots,
        extract_coordinates,
        calculate_snr,
        extract_signal,
        read_intensity,
        remove_duplicates,
        divide_main,
    )

    return {
        "tophat_spots": tophat_spots,
        "extract_coordinates": extract_coordinates,
        "calculate_snr": calculate_snr,
        "extract_signal": extract_signal,
        "read_intensity": read_intensity,
        "remove_duplicates": remove_duplicates,
        "divide_main": divide_main,
    }


# Provide delayed import interface
def __getattr__(name):
    """Delayed import of optional dependencies"""
    # Try traditional methods first (most commonly used)
    try:
        traditional_methods = _import_traditional_methods()
        if name in traditional_methods:
            return traditional_methods[name]
    except ImportError as e:
        pass

    # Try intensity extractors
    try:
        intensity_extractors = _import_intensity_extractors()
        if name in intensity_extractors:
            return intensity_extractors[name]
    except ImportError as e:
        pass

    # Try unified detector
    try:
        unified_detector = _import_unified_detector()
        if name in unified_detector:
            return unified_detector[name]
    except ImportError as e:
        pass

    # Try deep learning methods
    try:
        deep_learning_methods = _import_deep_learning_methods()
        if name in deep_learning_methods:
            return deep_learning_methods[name]
    except ImportError as e:
        pass

    # Try legacy functions
    try:
        legacy_functions = _import_legacy_functions()
        if name in legacy_functions:
            return legacy_functions[name]
    except ImportError as e:
        pass

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Core classes (always available)
    "SpotDetectionResult",
    "ImageProcessor",
    "TileProcessor",
    "MemmapImageLoader",
    "MultiChannelMemmapLoader",
    "estimate_memory_usage",
    # Configuration and channel management (always available)
    "load_config",
    "load_batch_config",
    "merge_configs",
    "load_spot_detection_config",
    "create_config_from_modules",
    "ChannelManager",
    "create_channel_manager_from_config",
    # Traditional methods (requires skimage - lazy loaded)
    "TophatBackgroundRemover",
    "LocalMaximaDetector",
    "DuplicateRemover",
    "SNRCalculator",
    "TraditionalSpotDetector",
    "MultiChannelTraditionalDetector",
    # Intensity extractors (requires scipy - lazy loaded)
    "DirectIntensityExtractor",
    "GaussianIntensityExtractor",
    "MaskIntensityExtractor",
    "IntegratedIntensityExtractor",
    "AdaptiveIntensityExtractor",
    "MultiScaleIntensityExtractor",
    "create_intensity_extractor",
    "create_preset_intensity_extractor",
    "INTENSITY_EXTRACTOR_CONFIGS",
    # Deep learning methods (requires stardist/tensorflow - lazy loaded)
    "StarDistDetector",
    "DeepLearningSpotDetector",
    "MultiChannelDeepLearningDetector",
    # Unified detector (requires all methods - lazy loaded)
    "UnifiedSpotDetector",
    "detect_spots",
    "create_detector_from_config",
    "create_preset_detector",
    "DEFAULT_CONFIGS",
    # Legacy functions (requires skimage - lazy loaded)
    "tophat_spots",
    "extract_coordinates",
    "calculate_snr",
    "extract_signal",
    "read_intensity",
    "remove_duplicates",
    "divide_main",
]
