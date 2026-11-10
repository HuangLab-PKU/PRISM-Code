"""Addressing a stitched channel by its per-channel filename, on every layout.

readout.yaml and segment_dapi.py name channels ``cyc_1_cy5.tif``. On a mosaic run that
name has to resolve to the same pixels inside mosaic.ome.tif / mosaic.ome.zarr, and on a
legacy run nothing about the old per-file behaviour may change.
"""
import importlib.util

import numpy as np
import pytest
import tifffile

from prism.readout import stitched
from prism.readout.stitched import (
    has_stitched, open_stitched_plane, parse_stitched_name, read_stitched,
    stitched_plane_shape,
)

from _stitched_layouts import CHANNELS, LAYOUTS, synthetic_planes, write_layout

zarr = pytest.importorskip("zarr")
pytest.importorskip("imagecodecs")

SHAPE = (200, 300)


@pytest.fixture(scope="module")
def planes():
    return synthetic_planes(SHAPE, n_spots=10)


@pytest.fixture(params=LAYOUTS)
def run(request, tmp_path, planes):
    return request.param, write_layout(tmp_path / request.param / "stitched",
                                       request.param, planes)


def test_parse_stitched_name():
    assert parse_stitched_name("cyc_1_cy5.tif") == (1, "cy5")
    assert parse_stitched_name("cyc_12_DAPI.tif") == (12, "DAPI")
    for bad in ("cy5.tif", "cyc_x_cy5.tif", "cycle_1_cy5.tif", "cyc_1.tif"):
        with pytest.raises(ValueError, match="cyc_<cycle>_<channel>"):
            parse_stitched_name(bad)


def test_every_layout_reads_the_same_pixels(run, planes):
    layout, d = run
    for chn in CHANNELS:
        fname = f"cyc_1_{chn}.tif"
        assert has_stitched(d, fname)
        assert stitched_plane_shape(d, fname) == SHAPE
        np.testing.assert_array_equal(read_stitched(d, fname), planes[chn],
                                      err_msg=f"{layout} {chn}")
        img = open_stitched_plane(d, fname)
        assert img.shape == SHAPE
        for y, x, h, w in [(0, 0, 64, 64), (150, 250, 128, 128), (37, 101, 50, 70)]:
            np.testing.assert_array_equal(np.asarray(img[y:y + h, x:x + w]),
                                          planes[chn][y:y + h, x:x + w],
                                          err_msg=f"{layout} {chn} block ({y},{x})")


def test_absent_channel(run):
    layout, d = run
    assert not has_stitched(d, "cyc_1_AF750.tif")
    assert not has_stitched(d, "cyc_2_cy5.tif")
    with pytest.raises(FileNotFoundError):
        open_stitched_plane(d, "cyc_2_cy5.tif")


def test_mosaic_handle_is_lazy(run):
    layout, d = run
    if layout == "tif":
        pytest.skip("per-channel TIFFs open as a memmap")
    assert not isinstance(open_stitched_plane(d, "cyc_1_cy5.tif"), np.ndarray)


def test_legacy_tif_behaviour_is_unchanged(tmp_path, planes):
    """Any filename works, and a 3-D file gives readout its first plane but segmentation
    the whole stack -- exactly what tifffile.memmap / tifffile.imread did before."""
    d = tmp_path / "stitched"
    d.mkdir()
    tifffile.imwrite(d / "DAPI_custom.tif", planes["DAPI"])
    stack = np.stack([planes["cy5"], planes["cy3"], planes["FAM"]])
    tifffile.imwrite(d / "cyc_1_cy5.tif", stack, photometric="minisblack")

    assert has_stitched(d, "DAPI_custom.tif")
    np.testing.assert_array_equal(read_stitched(d, "DAPI_custom.tif"), planes["DAPI"])
    np.testing.assert_array_equal(np.asarray(open_stitched_plane(d, "cyc_1_cy5.tif")),
                                  planes["cy5"])
    assert stitched_plane_shape(d, "cyc_1_cy5.tif") == SHAPE
    np.testing.assert_array_equal(read_stitched(d, "cyc_1_cy5.tif"), stack)


def test_mosaic_wins_over_leftover_tifs(tmp_path, planes):
    """A store can be re-stitched after conversion; per-channel TIFFs left beside it are
    then stale, so the mosaic is what gets read."""
    d = write_layout(tmp_path / "stitched", "ometiff", planes)
    tifffile.imwrite(d / "cyc_1_cy5.tif", np.zeros(SHAPE, np.uint16))
    np.testing.assert_array_equal(read_stitched(d, "cyc_1_cy5.tif"), planes["cy5"])


def test_unaddressable_name_on_a_mosaic_fails_loudly(tmp_path, planes):
    d = write_layout(tmp_path / "stitched", "ometiff", planes)
    with pytest.raises(ValueError, match="cyc_<cycle>_<channel>"):
        open_stitched_plane(d, "DAPI_custom.tif")


def test_3d_mosaic_is_refused_by_the_2d_readout(tmp_path):
    d = tmp_path / "stitched"
    d.mkdir()
    root = zarr.create_group(store=str(d / "mosaic.ome.zarr"), zarr_format=2, overwrite=True)
    root.create_array(name="0", shape=(1, 1, 3) + SHAPE, chunks=(1, 1, 1, 64, 64),
                      dtype="uint16")
    root.attrs["omero"] = {"channels": [{"label": "cy5"}]}
    root.attrs["spatial_img_core"] = {"cycles": [1], "channels": ["cy5"]}
    with pytest.raises(ValueError, match="expects 2-D"):
        open_stitched_plane(d, "cyc_1_cy5.tif")
    assert read_stitched(d, "cyc_1_cy5.tif").shape == (3,) + SHAPE


def test_missing_mosaic_dependency_names_the_extra(tmp_path, planes, monkeypatch):
    d = write_layout(tmp_path / "stitched", "ometiff", planes)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(stitched.importlib.util, "find_spec",
                        lambda name, *a: None if name == "imagecodecs"
                        else real_find_spec(name, *a))
    with pytest.raises(ModuleNotFoundError, match=r"prism\[mosaic\]"):
        open_stitched_plane(d, "cyc_1_cy5.tif")
