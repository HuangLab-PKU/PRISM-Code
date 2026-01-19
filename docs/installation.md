# PRISM 包安装说明

## 为什么需要专属环境？

**强烈建议创建专属的 conda 环境**，原因如下：

1. **依赖隔离**：避免与其他项目的依赖冲突（特别是 TensorFlow、NumPy 等版本敏感包）
2. **版本控制**：确保所有用户使用相同的依赖版本，保证结果可复现
3. **易于管理**：可以轻松删除和重建环境，不影响系统 Python
4. **团队协作**：通过 `environment.yml` 文件，团队成员可以创建完全相同的环境

---

## 方法1：使用 Conda（推荐）

### 步骤1：创建 conda 环境

进入 `code` 目录，使用 `environment.yml` 创建环境：

```bash
cd code
conda env create -f environment.yml
```

这会创建一个名为 `prism` 的 conda 环境，并安装所有依赖。

### 步骤2：激活环境

```bash
# Windows (PowerShell/CMD)
conda activate prism

# Linux/Mac
conda activate prism
```

### 步骤3：安装 PRISM 包（开发模式）

**重要**：PRISM 包需要单独安装，因为 `-e .` 需要在 `code/` 目录下运行：

```bash
# 确保在 code/ 目录下
cd code
pip install -e .
```

### 步骤4：验证安装

```python
python -c "from src.gene_calling.pipeline import SignalClassificationPipeline; print('✓ 安装成功')"
```

---

## 方法2：使用 pip + venv（备选）

如果你不想使用 conda，也可以使用 Python 的 venv：

### 步骤1：创建虚拟环境

```bash
cd code
python -m venv prism_env
```

### 步骤2：激活环境

```bash
# Windows (PowerShell)
.\prism_env\Scripts\Activate.ps1

# Windows (CMD)
prism_env\Scripts\activate.bat

# Linux/Mac
source prism_env/bin/activate
```

### 步骤3：安装依赖和包

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 日常使用

### 激活环境

每次使用前，记得激活环境：

```bash
conda activate prism
```

### 运行脚本

激活环境后，可以在任何目录运行：

```bash
# 方式1：直接运行（推荐）
python code/scripts/gene_calling.py
python code/scripts/readout.py

# 方式2：作为模块运行
cd code
python -m scripts.gene_calling
python -m scripts.readout
```

### 导入模块

```python
from src.gene_calling.pipeline import SignalClassificationPipeline
from src.readout.spot_detection import get_spot_coordinates
from src.image_process.utils.io_utils import get_tif_list
```

---

## 环境管理

### 更新环境

如果 `environment.yml` 有更新：

```bash
conda env update -f environment.yml --prune
```

### 导出当前环境

如果你想导出当前环境配置（用于分享或备份）：

```bash
conda env export > environment.yml
```

### 删除环境

如果环境出现问题，可以删除后重建：

```bash
conda deactivate  # 先退出环境
conda env remove -n prism
conda env create -f environment.yml  # 重新创建
```

### 查看已安装的包

```bash
conda list
pip list
```

---

## 开发模式的优势

使用 `pip install -e .` 安装后：

1. **代码修改立即生效**：修改 `src/` 下的代码后，无需重新安装，导入的包会使用最新代码
2. **保持代码结构**：不会复制文件到 site-packages，直接链接到源码目录
3. **IDE 支持更好**：可以跳转到源码、自动补全、类型检查
4. **标准做法**：符合 Python 包管理最佳实践

---

## 常见问题

### Q: 为什么 conda 和 pip 混用？

A: 这是标准做法：
- **Conda** 用于管理基础科学计算包（NumPy、SciPy、TensorFlow 等），这些包在 conda 中有预编译版本，兼容性更好
- **Pip** 用于安装 PRISM 包本身和 conda 中没有的包（如 `pathos`、`PySimpleGUI` 等）

### Q: 如何安装 GPU 版本的 TensorFlow？

A: 在 `environment.yml` 中，将：
```yaml
- tensorflow=2.16.1
```
改为：
```yaml
- tensorflow-gpu=2.16.1
```
或者创建环境后：
```bash
conda install -c conda-forge tensorflow-gpu=2.16.1
```

### Q: MATLAB Engine 如何安装？

A: MATLAB Engine 必须通过 MATLAB 的安装程序安装，不能通过 conda/pip：
```bash
cd "matlabroot/extern/engines/python"
python setup.py install
```

### Q: 环境激活后找不到命令？

A: 确保：
1. 已正确激活环境（命令行提示符前应显示 `(prism)`）
2. 已安装 PRISM 包：`pip install -e .`
3. 在正确的目录下运行脚本

---

## 卸载

### 卸载 PRISM 包

```bash
pip uninstall prism
```

### 删除整个环境

```bash
conda deactivate
conda env remove -n prism
```
