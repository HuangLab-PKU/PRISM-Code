"""
Memory-mapped image loader for efficient large image processing
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tifffile
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class MemmapImageLoader:
    """
    Memory-mapped image loader for efficient processing of large images

    This class provides memory-mapped access to large TIFF images, allowing
    efficient processing without loading entire images into memory.
    """

    def __init__(
        self, temp_dir: Optional[Union[str, Path]] = None, cleanup: bool = True
    ):
        """
        Initialize the memory-mapped image loader

        Args:
            temp_dir: Directory for temporary memory-mapped files
            cleanup: Whether to cleanup temporary files on exit
        """
        self.temp_dir = (
            Path(temp_dir)
            if temp_dir
            else Path(tempfile.gettempdir()) / "spot_detection_memmap"
        )
        self.cleanup = cleanup
        self.memmap_files: Dict[str, Path] = {}
        self.memmap_info: Dict[str, Dict] = {}

        # Create temp directory if it doesn't exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"MemmapImageLoader initialized with temp_dir: {self.temp_dir}")

    def create_memmap(
        self, image_path: Union[str, Path], key: str, force_recreate: bool = False
    ) -> Dict:
        """
        Create memory-mapped file for an image

        Args:
            image_path: Path to the TIFF image
            key: Unique key for this image
            force_recreate: Force recreation even if memmap exists

        Returns:
            Dictionary with memmap information
        """
        image_path = Path(image_path)
        memmap_path = self.temp_dir / f"{key}.dat"

        # Check if memmap already exists
        if memmap_path.exists() and not force_recreate:
            logger.info(f"Using existing memmap for {key}: {memmap_path}")
            return self.memmap_info[key]

        logger.info(f"Creating memmap for {key}: {image_path} -> {memmap_path}")

        # Load image to get shape and dtype
        with tifffile.TiffFile(image_path) as tif:
            image = tif.asarray()
            shape = image.shape
            dtype = image.dtype

        # Create memory-mapped file
        memmap_array = np.memmap(memmap_path, dtype=dtype, mode="w+", shape=shape)

        # Copy data to memmap
        memmap_array[:] = image[:]
        memmap_array.flush()
        del memmap_array

        # Store information
        self.memmap_files[key] = memmap_path
        self.memmap_info[key] = {
            "path": memmap_path,
            "shape": shape,
            "dtype": dtype,
            "original_path": image_path,
        }

        logger.info(f"Created memmap for {key}: shape={shape}, dtype={dtype}")
        return self.memmap_info[key]

    def load_tile(
        self, key: str, pad_x: int, pad_y: int, cut_x: int, cut_y: int
    ) -> np.ndarray:
        """
        Load a tile from memory-mapped image

        Args:
            key: Image key
            pad_x: X offset
            pad_y: Y offset
            cut_x: X size
            cut_y: Y size

        Returns:
            Image tile as numpy array
        """
        if key not in self.memmap_info:
            raise ValueError(f"No memmap found for key: {key}")

        info = self.memmap_info[key]
        memmap_path = info["path"]
        shape = info["shape"]
        dtype = info["dtype"]

        # Load memory-mapped array
        memmap_array = np.memmap(str(memmap_path), dtype=dtype, mode="r", shape=shape)

        # Extract tile
        if len(shape) == 3:
            # 3D image (Z, Y, X)
            tile = memmap_array[:, pad_y : pad_y + cut_y, pad_x : pad_x + cut_x]
        else:
            # 2D image (Y, X)
            tile = memmap_array[pad_y : pad_y + cut_y, pad_x : pad_x + cut_x]

        return tile.copy()  # Return a copy to avoid memmap issues

    def get_image_info(self, key: str) -> Dict:
        """Get information about a memory-mapped image"""
        if key not in self.memmap_info:
            raise ValueError(f"No memmap found for key: {key}")
        return self.memmap_info[key]

    def list_memmaps(self) -> Dict[str, Dict]:
        """List all created memory-mapped files"""
        return self.memmap_info.copy()

    def cleanup_memmap(self, key: str):
        """Clean up a specific memory-mapped file"""
        if key in self.memmap_files:
            memmap_path = self.memmap_files[key]
            if memmap_path.exists():
                memmap_path.unlink()
                logger.info(f"Cleaned up memmap: {memmap_path}")
            del self.memmap_files[key]
            del self.memmap_info[key]

    def cleanup_all(self):
        """Clean up all memory-mapped files"""
        for key in list(self.memmap_files.keys()):
            self.cleanup_memmap(key)

        # Remove temp directory if empty
        try:
            if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                self.temp_dir.rmdir()
                logger.info(f"Removed empty temp directory: {self.temp_dir}")
        except OSError:
            pass

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.cleanup:
            self.cleanup_all()


class MultiChannelMemmapLoader:
    """
    Multi-channel memory-mapped image loader

    Handles multiple channels with coordinated memory mapping
    """

    def __init__(
        self, temp_dir: Optional[Union[str, Path]] = None, cleanup: bool = True
    ):
        """
        Initialize multi-channel memory-mapped loader

        Args:
            temp_dir: Directory for temporary files
            cleanup: Whether to cleanup on exit
        """
        self.loader = MemmapImageLoader(temp_dir, cleanup)
        self.channel_info: Dict[str, Dict] = {}

    def create_channel_memmaps(
        self, image_dict: Dict[str, Union[str, Path]], progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Create memory-mapped files for multiple channels

        Args:
            image_dict: Dictionary mapping channel names to image paths
            progress: Whether to show progress bar

        Returns:
            Dictionary with memmap information for each channel
        """
        channels = list(image_dict.keys())
        if progress:
            channels = tqdm(channels, desc="Creating channel memmaps")

        for channel in channels:
            image_path = image_dict[channel]
            info = self.loader.create_memmap(image_path, channel)
            self.channel_info[channel] = info

        logger.info(f"Created memmaps for {len(image_dict)} channels")
        return self.channel_info

    def load_channel_tile(
        self, channel: str, pad_x: int, pad_y: int, cut_x: int, cut_y: int
    ) -> np.ndarray:
        """Load a tile for a specific channel"""
        return self.loader.load_tile(channel, pad_x, pad_y, cut_x, cut_y)

    def load_all_channel_tiles(
        self, pad_x: int, pad_y: int, cut_x: int, cut_y: int
    ) -> Dict[str, np.ndarray]:
        """Load tiles for all channels"""
        tile_dict = {}
        for channel in self.channel_info.keys():
            tile_dict[channel] = self.load_channel_tile(
                channel, pad_x, pad_y, cut_x, cut_y
            )
        return tile_dict

    def get_channel_info(self, channel: str) -> Dict:
        """Get information for a specific channel"""
        return self.loader.get_image_info(channel)

    def get_all_channel_info(self) -> Dict[str, Dict]:
        """Get information for all channels"""
        return self.channel_info.copy()

    def cleanup(self):
        """Clean up all memory-mapped files"""
        self.loader.cleanup_all()
        self.channel_info.clear()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


def estimate_memory_usage(
    image_shape: Tuple[int, ...], dtype: np.dtype, num_channels: int = 1
) -> Dict[str, float]:
    """
    Estimate memory usage for image processing

    Args:
        image_shape: Shape of the image
        dtype: Data type of the image
        num_channels: Number of channels

    Returns:
        Dictionary with memory usage estimates in GB
    """
    bytes_per_pixel = np.dtype(dtype).itemsize
    total_pixels = np.prod(image_shape)

    # Memory usage estimates
    original_memory = total_pixels * bytes_per_pixel * num_channels / (1024**3)
    processed_memory = original_memory * 2  # Original + processed
    memmap_memory = 0.1  # Minimal memory for memmap access

    return {
        "original_memory_gb": original_memory,
        "processed_memory_gb": processed_memory,
        "memmap_memory_gb": memmap_memory,
        "total_without_memmap_gb": original_memory + processed_memory,
        "total_with_memmap_gb": memmap_memory + processed_memory,
        "memory_savings_gb": original_memory - memmap_memory,
    }


# Example usage
if __name__ == "__main__":
    # Example of using the memory-mapped loader
    image_dict = {
        "ch1": "path/to/channel1.tif",
        "ch2": "path/to/channel2.tif",
        "ch3": "path/to/channel3.tif",
        "ch4": "path/to/channel4.tif",
    }

    with MultiChannelMemmapLoader() as loader:
        # Create memmaps for all channels
        loader.create_channel_memmaps(image_dict)

        # Load a tile from all channels
        tile_dict = loader.load_all_channel_tiles(
            pad_x=0, pad_y=0, cut_x=1000, cut_y=1000
        )

        print(f"Loaded tiles for {len(tile_dict)} channels")
        for channel, tile in tile_dict.items():
            print(f"{channel}: shape={tile.shape}, dtype={tile.dtype}")

