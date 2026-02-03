"""
Large-image block utilities: memmap-based block iteration for spot detection and readout.

Use image_memmap_blocks() to iterate over image blocks without loading the full image.
Each block can be passed to get_spot_coordinates() or read_intensity_tophat();
coordinates returned in block-local space can be converted to global with block_offset.
"""

from pathlib import Path
from typing import Union, Iterator, Tuple
import numpy as np
import logging

logger = logging.getLogger('readout_utils_image_blocks')

try:
    import tifffile
except ImportError:
    tifffile = None


def image_memmap_blocks(
    image_path: Union[str, Path],
    block_size: Tuple[int, int] = (2048, 2048),
    overlap: Tuple[int, int] = (64, 64),
    memmap: bool = True,
) -> Iterator[Tuple[Tuple[slice, slice], np.ndarray, Tuple[int, int]]]:
    """Yield image blocks from a TIFF path using memmap (or full read).

    For each block, yields (slices, block_array, offset) so that:
    - block_array is the 2D block (numpy array, possibly a view of memmap)
    - offset = (offset_y, offset_x) in global image coordinates; add to block-local
      (y, x) to get global coordinates.

    Parameters
    ----------
    image_path : str or Path
        Path to a 2D TIFF image.
    block_size : tuple (int, int)
        (height, width) of each block in pixels. Default (2048, 2048).
    overlap : tuple (int, int)
        (vertical, horizontal) overlap between adjacent blocks. Default (64, 64).
    memmap : bool
        If True, open with tifffile.memmap and yield slices (lazy load per block).
        If False, load full image with tifffile.imread and yield copies.

    Yields
    ------
    (slices, block, offset)
        slices : (slice_y, slice_x) for global image
        block : np.ndarray, 2D block (uint16)
        offset : (offset_y, offset_x) = (slice_y.start, slice_x.start)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))
    if tifffile is None:
        raise ImportError("tifffile is required for image_memmap_blocks")

    if memmap:
        img = tifffile.memmap(str(image_path))
    else:
        img = np.asarray(tifffile.imread(str(image_path)))

    if img.ndim != 2:
        # Multi-page or stack: take first frame
        if img.ndim == 3:
            img = img[0]
        else:
            raise ValueError(f"Expected 2D image, got ndim={img.ndim}")

    h, w = img.shape
    by, bx = block_size
    oy, ox = overlap
    step_y = max(1, by - oy)
    step_x = max(1, bx - ox)

    for start_y in range(0, h, step_y):
        for start_x in range(0, w, step_x):
            end_y = min(start_y + by, h)
            end_x = min(start_x + bx, w)
            sl_y = slice(start_y, end_y)
            sl_x = slice(start_x, end_x)
            block = np.asarray(img[sl_y, sl_x])
            offset = (int(start_y), int(start_x))
            yield (sl_y, sl_x), block, offset


def block_starts(
    height: int,
    width: int,
    block_size: Tuple[int, int] = (2048, 2048),
    overlap: Tuple[int, int] = (64, 64),
) -> list:
    """Return list of (start_y, start_x) for each block covering a grid of given size.

    Useful to build parallel tasks: one task per (start_y, start_x), worker opens
    memmap and reads that block.

    Parameters
    ----------
    height, width : int
        Image shape (H, W).
    block_size : tuple (int, int)
        (by, bx) block size.
    overlap : tuple (int, int)
        (oy, ox) overlap.

    Returns
    -------
    list of (int, int)
        [(start_y, start_x), ...] for each block.
    """
    by, bx = block_size
    oy, ox = overlap
    step_y = max(1, by - oy)
    step_x = max(1, bx - ox)
    out = []
    for start_y in range(0, height, step_y):
        for start_x in range(0, width, step_x):
            out.append((start_y, start_x))
    return out


def process_image_blocks(
    image_path: Union[str, Path],
    process_fn,
    block_size: Tuple[int, int] = (2048, 2048),
    overlap: Tuple[int, int] = (64, 64),
    memmap: bool = True,
    **process_kwargs,
) -> np.ndarray:
    """Run a per-block processor on an image and merge results with global coordinates.

    process_fn(block, **process_kwargs) must return (N, 2) array of (Y, X) in block-local
    coordinates. This function adds block offset so that returned coords are in global
    image coordinates, and concatenates all blocks (caller may deduplicate at boundaries).

    Parameters
    ----------
    image_path : str or Path
        Path to 2D TIFF.
    process_fn : callable
        Signature: process_fn(block: np.ndarray, **process_kwargs) -> np.ndarray
        Returns (N, 2) (Y, X) in block-local coordinates.
    block_size, overlap, memmap
        Passed to image_memmap_blocks().
    **process_kwargs
        Passed to process_fn(block, **process_kwargs).

    Returns
    -------
    np.ndarray
        (M, 2) array of global (Y, X) coordinates from all blocks.
    """
    all_coords = []
    for (_sl_y, _sl_x), block, (off_y, off_x) in image_memmap_blocks(
        image_path, block_size=block_size, overlap=overlap, memmap=memmap
    ):
        local = process_fn(block, **process_kwargs)
        if local is None or len(local) == 0:
            continue
        local = np.asarray(local)
        if local.ndim != 2 or local.shape[1] != 2:
            raise ValueError("process_fn must return (N, 2) array of (Y, X)")
        global_coords = local + np.array([off_y, off_x], dtype=local.dtype)
        all_coords.append(global_coords)
    if not all_coords:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(all_coords)


def with_memmap_blocks(
    block_size: Tuple[int, int] = (2048, 2048),
    overlap: Tuple[int, int] = (64, 64),
    memmap: bool = True,
):
    """Decorator: make a function that accepts (image_path, ...) run per block and merge coords.

    The decorated function must have signature:
        fn(image: np.ndarray, *args, **kwargs) -> np.ndarray
    and return (N, 2) (Y, X) in image-local coordinates.

    Usage:
        @with_memmap_blocks(block_size=(2048, 2048))
        def detect(image, method='tophat', **kwargs):
            return get_spot_coordinates(image, method=method, **kwargs)

        global_coords = detect(image_path, method='tophat', min_distance=2)
    """
    def decorator(fn):
        def wrapped(image_path, *args, **kwargs):
            return process_image_blocks(
                image_path,
                lambda block: fn(block, *args, **kwargs),
                block_size=block_size,
                overlap=overlap,
                memmap=memmap,
            )
        return wrapped
    return decorator
