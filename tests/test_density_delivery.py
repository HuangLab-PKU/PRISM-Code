import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from prism.density_delivery import publish_density


@pytest.mark.parametrize("probabilities, expected", [([.991, .99, np.nan], 1), ([.5, .99, np.nan], 0)])
def test_shared_thermal_delivery(tmp_path, probabilities, expected):
    pytest.importorskip("spatial_cells")
    from spatial_cells.maps.density_stack import thermal_colormap

    rd = tmp_path / "readout"
    rd.mkdir()
    stitched = tmp_path / "stitched"
    stitched.mkdir()
    tifffile.imwrite(stitched / "cyc_1_cy3.tif", np.ones((410, 410), np.uint16))
    pd.DataFrame(dict(index=[2, 5, 7], Y=[1, 205, 300], X=[10, 20, 30],
                      Gene=["A", "A", "B"], Probability=probabilities)).to_csv(
                          rd / "mapping_gmm.csv", index=False)
    publish_density(rd, "mapping_gmm.csv", ["A", "B"], "synthetic")
    assert json.loads((rd / "density_Q20_qc.json").read_text())["total_spots"] == expected
    assert tifffile.imread(rd / "density_Q20/A.tif").sum() == expected
    with tifffile.TiffFile(rd / "density_Q20_all.tif") as tif:
        assert tif.imagej_metadata["Labels"] == ["A.tif", "B.tif"]
        assert np.array_equal(tif.pages[0].colormap, thermal_colormap())
