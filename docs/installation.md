# Installation

PRISM is a standard PEP 621 Python package; all dependencies are declared in `pyproject.toml`. It requires **Python ≥ 3.10**.

## 1. Install the package

```bash
git clone https://github.com/HuangLab-PKU/PRISM-Code
cd PRISM-Code
pip install -e .
```

We recommend a dedicated environment (conda/mamba shown; a `venv` works too):

```bash
mamba create -n prism python=3.11
mamba activate prism
pip install -e .
```

Verify:

```bash
python -c "from prism.gene_calling.pipeline import SignalClassificationPipeline; from prism.readout.spot_detection import get_spot_coordinates; print('ok')"
```

The baseline covers intensity readout, the **GMM** and **codebook-GMM** gene-calling methods, and DAPI segmentation via the cellpose path.

## 2. Spot-detection backend (Spotiflow)

The default `detection_method: spotiflow` in `configs/readout.yaml` needs **Spotiflow**, which ships as an optional extra:

```bash
pip install -e ".[spotiflow]"
```

`scripts/readout.py` runs Spotiflow with `device='cuda'`, so the default detection path needs a CUDA GPU. Without a GPU (or if you skip the install), set `detection_method: gaussian_tophat` (or `tophat`) in `configs/readout.yaml` to use the traditional detectors — no extra install or GPU needed.

## 3. Optional extras

| Extra | Purpose | Install |
|---|---|---|
| `spotiflow` | Default deep-learning spot detector (needs a CUDA GPU) | `pip install -e ".[spotiflow]"` |
| `cellpose` | PyTorch-based cell segmentation | `pip install -e ".[cellpose]"` |
| `postcode` | **Experimental** Bayesian decoding (pyro + PyTorch) | `pip install -e ".[postcode]"` + vendored PoSTcode (below) |
| `stardist-tf` | StarDist 2D/3D segmentation (TensorFlow 2.10) | separate environment (below) |
| `gui` | Legacy PySimpleGUI (manual thresholding) | `pip install -e ".[gui]"` |
| `test` | pytest | `pip install -e ".[test]"` |

### PoSTcode (experimental, HuangLab fork)

PoSTcode is **not on PyPI**. It is vendored in the public SPRINTseq repository; install it editable from there:

```bash
git clone https://github.com/HuangLab-PKU/SPRINTseq
pip install -e ".[postcode]"
pip install -e SPRINTseq/experiments/src/postcode
python -c "import postcode; print('ok')"
```

PoSTcode is an **opt-in, experimental** alternative to the default GMM methods. (`pip install postcode` would install an unrelated package — use the vendored tree.)

### StarDist environment (TensorFlow 2.10)

StarDist still ships TensorFlow 2.10 wheels, which pin `numpy < 2` and conflict with the NumPy 2.x baseline. Build StarDist in a **separate** environment:

```bash
mamba create -n prism-stardist python=3.10
mamba activate prism-stardist
cd PRISM-Code
pip install -e ".[stardist-tf]"
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

On Windows, TF 2.10 is the last version with native GPU support; GPU StarDist needs matching **CUDA 11.2 / cuDNN 8.1**. Other environments fall back to CPU StarDist (much slower).

## 4. Upstream image processing (`spatial_img_core`)

PRISM starts from stitched images. If you need the raw-image → stitched-image steps (focal stacking, illumination correction, registration, stitching, AIRLOCALIZE), install the companion `spatial_img_core` package in the same environment. It is **not yet public** — request access from the authors at **huanglab111@gmail.com**. Its MATLAB-backed stitching / focal-stack backends are optional and require the `matlabengine` wheel only if you use them.

## 5. Everyday use

After activating the environment, run scripts from the repository root:

```bash
python scripts/readout.py <RUN_ID>
python scripts/gene_calling.py
python scripts/segment_dapi.py <RUN_ID>
```

The `prism` package imports from any working directory (the editable install resolves via site-packages):

```python
from prism.gene_calling.pipeline import SignalClassificationPipeline
from prism.readout.spot_detection import get_spot_coordinates
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'prism'`
- Confirm the correct environment is active.
- Confirm `pip install -e .` succeeded: `pip show prism` should point at your clone.

### `Spotiflow is not installed`
- Run `pip install spotiflow`, or switch `detection_method` to `gaussian_tophat` / `tophat` in `configs/readout.yaml`.

### `import postcode` fails
- PoSTcode must be installed editable from the SPRINTseq vendored tree (see above).

### StarDist can't find a GPU on Windows
- Only an environment with TF 2.10 + matching CUDA 11.2 / cuDNN 8.1 has GPU StarDist; other environments run CPU-only.

### Do I reinstall after editing code?
- No. `pip install -e .` is an editable install; edits under `prism/` take effect immediately. Re-run it only after changing `pyproject.toml` dependencies, scripts, or package configuration.

## Uninstall

```bash
pip uninstall prism
```
