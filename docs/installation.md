# PRISM 包安装说明

PRISM 的 `code/` 是一个标准的 PEP 621 Python 包，所有依赖通过 `pyproject.toml` 声明。本文档介绍两种推荐的安装方式，以及可选的 extras。

---

## 方法 1：装到现有的 `spatial-prep-dp` 环境（推荐）

实验室 spatial 流水线（PRISM / SPRINTseq / spatial_img_core / postcode）已经 editable-installed 在 `spatial-prep-dp` 这个 mamba env 里，这是日常工作的首选。

```powershell
mamba activate spatial-prep-dp
cd C:\Users\Mingchuan\Huanglab\PRISM\code
pip install -e .
```

验证：

```powershell
python -c "from prism.gene_calling.pipeline import SignalClassificationPipeline; from prism.readout.spot_detection import get_spot_coordinates; print('ok')"
```

`spatial-prep-dp` 提供的 baseline 已经覆盖 PRISM 的所有基础依赖（NumPy、SciPy、scikit-image、scikit-learn、OpenCV、tifffile、matplotlib、seaborn、cmap、pathos、csbdeep、PyTorch、cellpose 等），无需再装。

---

## 方法 2：单建一个轻量的 `prism` 环境

如果你希望把 PRISM 跟其他项目隔离，可以单建一个干净的 env：

```bash
mamba create -n prism python=3.11
mamba activate prism
cd code
pip install -e .
```

这会按 `pyproject.toml` 里 `dependencies` 列出的 baseline 装齐。Optional extras 见下。

---

## 可选 extras

| Extra | 用途 | 安装 |
|---|---|---|
| `postcode` | Bayesian 解码（基于 pyro + PyTorch）；用于多轮 SBS 场景 | `pip install -e ".[postcode]"` 后**额外**装 vendored postcode（见下） |
| `cellpose` | 基于 PyTorch 的细胞分割（推荐路径） | `pip install -e ".[cellpose]"` |
| `stardist-tf` | StarDist 2D/3D 分割；依赖 TensorFlow 2.10（Windows 上 GPU 最后一版） | **必须单独 env**，见 [StarDist 环境](#stardist-环境windows-gpu) |
| `gui` | PySimpleGUI 老式 GUI（manual thresholding） | `pip install -e ".[gui]"` |
| `test` | pytest | `pip install -e ".[test]"` |

### PoSTcode（HuangLab fork）

PoSTcode 不在 PyPI 上，需要从 SPRINTseq 仓库的 vendored 路径 editable-install：

```powershell
mamba activate spatial-prep-dp
pip install -e C:\Users\Mingchuan\Huanglab\SPRINTseq\experiments\src\postcode
python -c "import postcode; print('ok')"
```

### StarDist 环境（Windows GPU）

StarDist 仍依赖 TensorFlow 2.10 wheels，且 Windows 上的 TF 2.10 把 NumPy 钉在 < 2.0，跟 `spatial-prep-dp` 的 NumPy 2.x 冲突。**所以 StarDist 必须用单独的 env**。仓库里已经维护了 `prism-seg-gpu`（Python 3.10 + CUDA 11.2 + cuDNN 8.1 + TF 2.10）：

```powershell
mamba activate prism-seg-gpu
cd C:\Users\Mingchuan\Huanglab\PRISM\code
pip install -e ".[stardist-tf]"
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

应看到 `GPUs: [PhysicalDevice(name='/physical_device:GPU:0', ...)]`。如果分割脚本启动时 `prism.cell_segmentation.unified_segmentation` 打印 `TensorFlow: using GPU(s) [...]`，GPU 就绪。

---

## 上游图像处理（spatial_img_core）

PRISM 从 stitched 图开始；如果你需要从原始小图到 stitched 的完整链路（focal stacking、illumination correction、registration、stitching），请在同一 mamba env 里把 sibling 包 `spatial_img_core` 也 editable 装上：

```powershell
mamba activate spatial-prep-dp
pip install -e C:\Users\Mingchuan\Huanglab\spatial_img_core\core
spatial-img-pipeline --help
```

`spatial_img_core` 自带 BaSiCPy / 传统 CIDRE 光照校正、GPU phase-correlation 配准、pcorr_bigstitcher / MIST 拼接、focal stacking 等多套后端。其 MATLAB-backed 后端可选；如使用，再按 MATLAB 官方步骤装 `matlabengine`（PyPI 上的 wheel 已可用，无需手动 `python setup.py install`）。

---

## 日常使用

激活环境后，可以从任何目录运行脚本：

```powershell
mamba activate spatial-prep-dp
python C:\Users\Mingchuan\Huanglab\PRISM\code\scripts\gene_calling.py
python C:\Users\Mingchuan\Huanglab\PRISM\code\scripts\segment_dapi.py <run_id>
```

`prism` 包可以在任何 cwd 下 import（editable install 通过 site-packages 解析，不依赖工作目录）：

```python
from prism.gene_calling.pipeline import SignalClassificationPipeline
from prism.readout.spot_detection import get_spot_coordinates

# Image-side helpers come from the sibling spatial_img_core package:
from spatial_img_core.utils.io_utils import get_tif_list
```

**注意**：不要把 Python 的 cwd 设在 `C:\Users\Mingchuan\Huanglab\`（即 Huanglab workspace 根目录）。该目录下有 `spatial_img_core/` 这个跟同名包冲突的子目录，PEP 420 namespace package 机制会先找到目录、屏蔽 editable install。`PRISM/` 跟 `prism` 因大小写不同不冲突，但保持 `cd code/` 或绝对路径运行更稳。

---

## 常见问题

### Q: `import prism` 报错 `ModuleNotFoundError`？

确认：

1. 已激活正确 env：`mamba env list` 看 `*` 在 `spatial-prep-dp`（或 `prism`、`prism-seg-gpu`）那一行。
2. 已 `pip install -e .`：`pip show prism` 看 Location 是否指向 `C:\Users\Mingchuan\Huanglab\PRISM\code`。
3. 没有从 `Huanglab/` 根目录直接 `python ...` 运行（见上）。

### Q: `import postcode` 报错？

PoSTcode 必须从 SPRINTseq 的 vendored 路径 editable-install（见上）。`pip install postcode` 会装到一个无关的同名包。

### Q: StarDist 在 Windows 上找不到 GPU？

只有 `prism-seg-gpu` 这个 env 里有匹配 TF 2.10 的 CUDA 11.2 / cuDNN 8.1。其它 env（`spatial-prep-dp`、自建 `prism`）只能跑 CPU StarDist，速度差很多。优先用 `prism-seg-gpu`。

### Q: 改了代码要不要 reinstall？

不需要。`pip install -e .` 是 editable install，编辑 `prism/` 下任意 .py 立即生效。只有改 `pyproject.toml` 的依赖、scripts 或 packages 配置才需要重跑 `pip install -e .`。

---

## 卸载

```powershell
mamba activate <env>
pip uninstall prism
```

要删整个 env：

```powershell
mamba deactivate
mamba env remove -n prism   # 或 prism-seg-gpu
```

注意：`spatial-prep-dp` 是工作区共享 env，不要随意 remove。
