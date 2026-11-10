"""Open a stitched channel by its per-channel filename, whatever the layout on disk.

``readout.yaml`` and ``segment_dapi.py`` name channels by the TIFF the stitcher used to
write per channel (``cyc_1_cy5.tif``, ``cyc_1_DAPI.tif``). Current stitching output is a
single ``mosaic.ome.tif`` (finalized) or ``mosaic.ome.zarr`` (still being written) holding
every channel; for those the filename only says which ``(cycle, channel)`` to address, so
existing configs keep working unchanged. Layout detection and precedence are
:func:`prism.readout.mosaic.backend`'s: a mosaic wins over per-channel TIFFs left beside it.
"""

import importlib.util
from pathlib import Path

import numpy as np
import tifffile

from .mosaic import backend, has_mosaic, mosaic_shape, open_mosaic

__all__ = ["parse_stitched_name", "has_stitched", "open_stitched_plane",
           "stitched_plane_shape", "read_stitched"]

# Reading pixels out of a mosaic needs these; the legacy TIFF path needs neither.
_MOSAIC_DEPS = {"zarr": ("zarr",), "ometiff": ("zarr", "imagecodecs")}


def parse_stitched_name(fname: str) -> tuple[int, str]:
    """``'cyc_1_cy5.tif'`` -> ``(1, 'cy5')``.

    Raises:
        ValueError: If the name is not ``cyc_<cycle>_<channel>.tif``.
    """
    parts = Path(fname).stem.split("_", 2)
    if len(parts) != 3 or parts[0] != "cyc" or not parts[1].isdigit():
        raise ValueError(
            f"Cannot address {fname!r} inside a stitched mosaic; channel files must be "
            "named cyc_<cycle>_<channel>.tif (e.g. cyc_1_cy5.tif)"
        )
    return int(parts[1]), parts[2]


def _require_mosaic_deps(stitch_dir: Path) -> None:
    missing = [m for m in _MOSAIC_DEPS.get(backend(stitch_dir), ())
               if importlib.util.find_spec(m) is None]
    if missing:
        raise ModuleNotFoundError(
            f"{stitch_dir} holds a stitched mosaic ({backend(stitch_dir)}); reading it "
            f"needs {', '.join(missing)}: pip install \"prism[mosaic]\""
        )


def has_stitched(stitch_dir, fname: str) -> bool:
    """Whether the channel ``fname`` names is present, on either layout."""
    stitch_dir = Path(stitch_dir)
    if backend(stitch_dir) == "tif":
        return (stitch_dir / fname).is_file()
    return has_mosaic(stitch_dir, *parse_stitched_name(fname))


def open_stitched_plane(stitch_dir, fname: str):
    """2-D handle sliced ``a[y0:y1, x0:x1]`` by the readout block loops, read lazily.

    A per-channel TIFF opens as a memmap and keeps readout's old handling of a 3-D file
    (first plane). A mosaic has to be 2-D: readout is a 2-D pipeline, and silently
    taking one z-plane of a 3-D mosaic would pass off a partial read as the whole.

    Raises:
        FileNotFoundError: If the channel is absent.
        ValueError: If a mosaic channel is not 2-D.
    """
    stitch_dir = Path(stitch_dir)
    if backend(stitch_dir) == "tif":
        img = tifffile.memmap(str(stitch_dir / fname))
        return img[0] if img.ndim == 3 else img
    _require_mosaic_deps(stitch_dir)
    img = open_mosaic(stitch_dir, *parse_stitched_name(fname))
    if img.ndim != 2:
        raise ValueError(
            f"{fname} in {stitch_dir} is {img.ndim}-D {img.shape}; readout expects 2-D"
        )
    return img


def stitched_plane_shape(stitch_dir, fname: str) -> tuple[int, int]:
    """``(h, w)`` of the plane :func:`open_stitched_plane` returns, without reading pixels."""
    stitch_dir = Path(stitch_dir)
    if backend(stitch_dir) == "tif":
        with tifffile.TiffFile(stitch_dir / fname) as t:
            sh = t.pages[0].shape
        return (sh[0], sh[1]) if len(sh) == 2 else (sh[1], sh[2])
    return tuple(mosaic_shape(stitch_dir, *parse_stitched_name(fname)))


def read_stitched(stitch_dir, fname: str) -> np.ndarray:
    """The whole image, in memory and in its native dimensionality (2-D or a z-stack).

    For consumers that need all of it at once, such as nuclear segmentation.
    """
    stitch_dir = Path(stitch_dir)
    if backend(stitch_dir) == "tif":
        return tifffile.imread(stitch_dir / fname)
    _require_mosaic_deps(stitch_dir)
    return np.asarray(open_mosaic(stitch_dir, *parse_stitched_name(fname)))
