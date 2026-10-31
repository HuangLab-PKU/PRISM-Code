"""scripts/readout.py gives the same answer whichever stitched layout holds the pixels.

The readout only ever sees blocks sliced out of a per-channel handle, so the same pixels
stored as per-channel TIFFs, an OME-Zarr store or a finalized OME-TIFF must produce
identical position.csv / intensity.csv. Uses the tophat detector, so neither spotiflow
nor a GPU is needed; blocks are small so the grid has several rows, columns and ragged
edges.
"""
import importlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from _stitched_layouts import LAYOUTS, synthetic_planes, write_layout

pytest.importorskip("zarr")
pytest.importorskip("imagecodecs")

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SPOT_FILES = ["cyc_1_cy5.tif", "cyc_1_TxRed.tif", "cyc_1_cy3.tif", "cyc_1_FAM.tif"]
RUN_ID = "20990101_TEST_layouts"


@pytest.fixture(scope="module")
def readout_script():
    # Imported by name from a sys.path entry (not exec'd from a file path): the block
    # workers run in spawned processes, which must be able to import it again.
    mp = pytest.MonkeyPatch()
    mp.syspath_prepend(str(SCRIPTS_DIR))
    try:
        yield importlib.import_module("readout")
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def config_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("config")
    cfg = {
        "base_dir": str(d),                   # unused: main() takes the dirs directly
        "channel_files": SPOT_FILES,
        "detection_method": "tophat",
        "detection_snr": {Path(f).stem: 3.0 for f in SPOT_FILES},
        "tophat_radius": 3,
        "search_radius": 1,
        "dedup_threshold": 2,
        "block_size": [128, 256],
        "block_overlap": [16, 16],
        "n_workers": 2,
    }
    (d / "readout.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return d


def _run(readout, config_dir, root, layout, planes):
    src = root / layout / f"{RUN_ID}_processed"
    stc_dir = write_layout(src / "stitched", layout, planes)
    read_dir = src / "readout"
    read_dir.mkdir()
    readout.load_config(config_dir)
    readout.main(run_id=RUN_ID, stc_dir=stc_dir, read_dir=read_dir)
    position = pd.read_csv(read_dir / "position.csv")
    intensity = pd.read_csv(read_dir / "intensity.csv")
    merged = position.merge(intensity, on="index").drop(columns="index")
    return merged.sort_values(["Y", "X"]).reset_index(drop=True)


def test_readout_output_is_identical_across_layouts(readout_script, config_dir, tmp_path):
    planes = synthetic_planes((300, 1100), n_spots=80)
    out = {layout: _run(readout_script, config_dir, tmp_path, layout, planes)
           for layout in LAYOUTS}

    ref = out["tif"]
    assert list(ref.columns) == ["Y", "X"] + [Path(f).stem for f in SPOT_FILES]
    # 80 planted spots per channel; the tophat threshold also admits noise peaks, so
    # this only guards against comparing empty outputs
    assert len(ref) > 250, len(ref)
    for layout in ("zarr", "ometiff"):
        pd.testing.assert_frame_equal(out[layout], ref, obj=f"{layout} vs tif readout")
