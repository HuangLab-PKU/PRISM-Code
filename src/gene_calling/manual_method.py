"""
Manual methods for post-processing, e.g., mask-based relabeling.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm


class ManualThresholdAdjuster:
    """
    Manual threshold adjustment functionality from legacy code.

    Provides methods for manual relabeling using mask images.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Use unified color dimension naming
        self.plot_columns = config.get("plot_columns", ["color_dim_1", "color_dim_2"])
        self.xrange = config.get("xrange", [-0.8, 0.8])
        self.yrange = config.get("yrange", [-0.6, 0.8])
        self.num_per_layer = config.get("num_per_layer", 15)

    def relabel_mask(
        self,
        intensity: pd.DataFrame,
        mask: np.ndarray,
        mode: str = "replace",
        ori_label: int = None,
        ch_label: int = None,
        G_layer: int = None,
    ) -> pd.DataFrame:
        """Relabel data points based on mask image."""
        intensity_tmp = intensity.copy()
        x, y = mask.shape[0], mask.shape[1]

        if mode == "replace":
            if G_layer is None:
                data = intensity[
                    intensity["G_layer"] == (ori_label - 1) // self.num_per_layer
                ]
            else:
                data = intensity[intensity["G_layer"] == G_layer]
        elif mode == "discard":
            data = intensity[intensity["label"] == ori_label]
            data["label"] = [-1] * len(data)

        # Convert coordinates to mask indices (plot_columns are color_dim_1/2)
        data["x"] = (
            (self.yrange[1] - data[self.plot_columns[1]])
            * x
            / (self.yrange[1] - self.yrange[0])
        )
        data["y"] = (
            (data[self.plot_columns[0]] - self.xrange[0])
            * y
            / (self.xrange[1] - self.xrange[0])
        )
        data["x"] = data["x"].astype(int)
        data["y"] = data["y"].astype(int)

        # Apply mask
        mask_values = mask[data["x"].values, data["y"].values]
        data.loc[mask_values, "label"] = ch_label
        intensity_tmp.loc[data.index, "label"] = data["label"]
        intensity_tmp = intensity_tmp[intensity_tmp["label"] != -1]

        return intensity_tmp

    def relabel(self, intensity: pd.DataFrame, mask_dir: str, mode: str = "discard") -> pd.DataFrame:
        """Relabel data using multiple mask images."""
        # Find mask files
        re_label = [
            match.group(1)
            for filename in os.listdir(mask_dir)
            if (match := re.search(r"mask_(\d+)\.png$", filename))
        ]

        if len(re_label) == 0:
            return intensity

        intensity_relabel = intensity.copy()

        for label in tqdm(sorted(list(map(int, re_label))), desc=f"Relabeling, mode={mode}"):
            mask_path = os.path.join(mask_dir, f"mask_{label}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(bool)

            intensity_relabel = self.relabel_mask(
                intensity_relabel, mask=mask, ori_label=label, ch_label=label, mode=mode
            )

        return intensity_relabel


