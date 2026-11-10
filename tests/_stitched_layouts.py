"""The same PRISM channels written in each stitched layout readout has to accept.

Mirrors a finalized PRISM run: one OME Image, cycle 1, channels cy5/TxRed/cy3/FAM/DAPI,
level 0 plus one 2x SubIFD level, the store attrs as JSON in the Image Description.
"""
import json

import numpy as np
import tifffile

CHANNELS = ["cy5", "TxRed", "cy3", "FAM", "DAPI"]
CYCLE = 1
LAYOUTS = ("tif", "zarr", "ometiff")


def synthetic_planes(shape, n_spots=80, seed=0):
    """Noisy background plus Gaussian spots, different in every channel."""
    rng = np.random.default_rng(seed)
    planes = {}
    for chn in CHANNELS:
        img = rng.normal(200, 20, size=shape)
        centres = rng.uniform([6, 6], [shape[0] - 6, shape[1] - 6], size=(n_spots, 2))
        for y, x in centres:
            y0, x0 = int(y) - 5, int(x) - 5
            yy, xx = np.mgrid[y0:y0 + 11, x0:x0 + 11]
            img[y0:y0 + 11, x0:x0 + 11] += rng.uniform(1500, 4000) * np.exp(
                -((yy - y) ** 2 + (xx - x) ** 2) / 3.0)
        planes[chn] = np.clip(img, 0, 65535).astype(np.uint16)
    return planes


def _stack(planes):
    return np.stack([planes[c] for c in CHANNELS])[None]      # (t=1, c, y, x)


def _attrs():
    return {"multiscales": [{"version": "0.4", "name": "spots",
                             "axes": [{"name": a} for a in "tcyx"],
                             "datasets": [{"path": "0"}]}],
            "omero": {"channels": [{"label": c} for c in CHANNELS]},
            "spatial_img_core": {"cycles": [CYCLE], "channels": CHANNELS, "missing": []}}


def write_layout(stitch_dir, layout, planes):
    """Write ``planes`` ({channel: 2-D array}) into ``stitch_dir`` in ``layout``."""
    stitch_dir.mkdir(parents=True, exist_ok=True)
    if layout == "tif":
        for chn, img in planes.items():
            tifffile.imwrite(stitch_dir / f"cyc_{CYCLE}_{chn}.tif", img)
    elif layout == "zarr":
        import zarr

        data = _stack(planes)
        root = zarr.create_group(store=str(stitch_dir / "mosaic.ome.zarr"),
                                 zarr_format=2, overwrite=True)
        arr = root.create_array(name="0", shape=data.shape, chunks=(1, 1, 64, 64),
                                dtype=data.dtype)
        arr[:] = data
        root.attrs.update(_attrs())
    elif layout == "ometiff":
        data, attrs = _stack(planes), _attrs()
        opts = dict(tile=(256, 256), compression="zstd", predictor=True,
                    photometric="minisblack")
        meta = {"Name": "spots", "axes": "TCYX", "Channel": {"Name": CHANNELS},
                "Description": json.dumps({"spatial_img_core_store": {"attrs": attrs}})}
        with tifffile.TiffWriter(stitch_dir / "mosaic.ome.tif", bigtiff=True, ome=True) as tif:
            tif.write(data, subifds=1, metadata=meta, **opts)
            tif.write(data[..., ::2, ::2], subfiletype=1, **opts)
    else:
        raise ValueError(layout)
    return stitch_dir
