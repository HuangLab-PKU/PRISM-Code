"""Publish the standard spatial-cells Q20 products for a completed mapping."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os

import numpy as np
import pandas as pd


def publish_density(readout, mapping, genes, run_id, *, stitched=None):
    from .readout.mosaic import list_mosaics
    readout = Path(readout)
    stitched = Path(stitched) if stitched is not None else readout.parent / "stitched"
    if not stitched.is_dir() or not list_mosaics(stitched):
        return  # Imported tables can be classified without a spatial canvas.
    from spatial_cells.maps.density import generate_density_maps, get_stitched_shape
    from spatial_cells.maps.density_stack import build_density_stack
    from spatial_cells.qc import generate_density_qc
    from tifffile import imwrite

    genes = sorted(set(genes))
    with TemporaryDirectory(prefix=".density-", dir=readout.parent) as temporary:
        stage = Path(temporary)
        # Project only qualifying rows for the in-memory spatial-cells reader.
        with (stage / "position.csv").open("w", newline="") as pos, (
            stage / "mapping_postcode.csv"
        ).open("w", newline="") as calls:
            rows = 0
            for i, frame in enumerate(pd.read_csv(
                readout / mapping, chunksize=250000,
                usecols=["index", "Y", "X", "Gene", "Probability"],
            )):
                kept = frame[frame.Gene.isin(genes) & (frame.Probability > 0.99)]
                kept[["index", "Y", "X"]].to_csv(pos, index=False, header=i == 0)
                kept[["index", "Gene", "Probability"]].to_csv(calls, index=False, header=i == 0)
                rows += len(kept)
        panel = stage / "panel.csv"
        pd.DataFrame({"Gene": genes}).to_csv(panel, index=False)
        maps = stage / "density_Q20"
        if rows:
            cube, names, filtered = generate_density_maps(
                stage, stitched, maps, threshold=.99, fac=200, ref_file=panel)
        else:
            maps.mkdir()
            h, w = get_stitched_shape(stitched)
            cube = np.zeros((len(genes), h // 200 + 1, w // 200 + 1), dtype=np.uint16)
            names, filtered = genes, pd.DataFrame()
            for gene, plane in zip(genes, cube):
                imwrite(maps / f"{gene}.tif", plane)
        if int(cube.sum(dtype=np.uint64)) != rows:
            raise ValueError("Density counts do not match qualifying mapping rows")
        generate_density_qc(filtered, cube, names, stage, run_id, .99, 200, "Q20")
        build_density_stack(maps, genes, stage / "density_Q20_all.tif")
        destination = readout / maps.name
        destination.mkdir(exist_ok=True)
        if {p.name for p in destination.glob("*.tif")} - {p.name for p in maps.glob("*.tif")}:
            raise ValueError("Existing density maps belong to another panel")
        for source in maps.glob("*.tif"):
            os.replace(source, destination / source.name)
        for name in ("density_Q20_all.tif", "density_Q20_qc.png", "density_Q20_qc.json"):
            os.replace(stage / name, readout / name)
