# PoSTcode方法实现方案

## 一、概述

本方案将PoSTcode概率解码方法集成到PRISM的gene calling框架中，作为`gene_calling.methods`下的一个新方法。

### 核心设计原则
1. **避免可识别性陷阱**：采用"P99粗校正 + PoSTcode微调"的两阶段策略
2. **分通道缩放**：修改PoSTcode的`codes_tr_v`为通道特异性参数
3. **背景建模**：背景code的前4维（比例）固定为0，仅第5维（强度）可学习
4. **输出格式**：保留所有类别的概率 + 熵值（参考SPRINTseq）

---

## 二、技术架构

### 2.1 文件结构

```
code/src/gene_calling/
├── methods/
│   ├── __init__.py
│   ├── gmm_method.py          # 现有GMM方法
│   └── postcode_method.py     # 新增PoSTcode方法
├── utils/
│   ├── generate_prism_codebook.py  # Codebook生成工具
│   └── postcode_wrapper.py         # PoSTcode封装（可选）
└── configs/
    └── codebooks/
        └── prism30_codebook.csv    # PRISM30 codebook
```

### 2.2 方法接口

`PostcodeMethod`类需要实现与`GMMMethod`相同的接口：
- `__init__(config)`: 初始化
- `preprocess(data)`: 数据预处理（P99缩放 + 比例+强度构造）
- `fit(data)`: 训练PoSTcode模型
- `predict(data)`: 预测并返回`ClassificationResult`
- `extract_features(data)`: 特征提取（用于兼容性，实际不需要）

---

## 三、实现步骤

### 步骤1：数据预处理（`preprocess`方法）

#### 1.1 P99分位数粗校正

```python
def _p99_channel_scaling(self, intensity_df: pd.DataFrame) -> tuple:
    """
    计算每个通道的P99分位数，并以ch1为基准进行缩放。
    
    返回：
    - scaled_df: 缩放后的强度DataFrame
    - scale_factors: 缩放因子字典
    """
    # 计算P99分位数
    p99_values = {
        'ch1': np.percentile(intensity_df['ch1'], 99),
        'ch2': np.percentile(intensity_df['ch2'], 99),
        'ch3': np.percentile(intensity_df['ch3'], 99),
        'ch4': np.percentile(intensity_df['ch4'], 99),
    }
    
    # 以ch1为基准
    baseline = p99_values['ch1']
    scale_factors = {
        'ch1': 1.0,
        'ch2': baseline / p99_values['ch2'] if p99_values['ch2'] > 0 else 1.0,
        'ch3': baseline / p99_values['ch3'] if p99_values['ch3'] > 0 else 1.0,
        'ch4': baseline / p99_values['ch4'] if p99_values['ch4'] > 0 else 1.0,
    }
    
    # 应用缩放
    scaled_df = intensity_df.copy()
    for ch, factor in scale_factors.items():
        scaled_df[ch] = scaled_df[ch] * factor
    
    return scaled_df, scale_factors
```

#### 1.2 构造观测向量（比例 + 强度）

```python
def _construct_observation_vector(self, scaled_df: pd.DataFrame) -> np.ndarray:
    """
    构造5维观测向量：[ch1/A, ch2/A, ch3/A, ch4/A, log10(sum_all)]
    
    返回：numpy array (N, 5, 1) 用于PoSTcode
    """
    # 计算比例（A = ch1 + ch2 + ch4，不包括ch3）
    A = scaled_df['ch1'] + scaled_df['ch2'] + scaled_df['ch4']
    A = A.replace(0, 1e-10)  # 避免除零
    
    # 计算比例
    ratios = pd.DataFrame({
        'ch1/A': scaled_df['ch1'] / A,
        'ch2/A': scaled_df['ch2'] / A,
        'ch3/A': scaled_df['ch3'] / A,
        'ch4/A': scaled_df['ch4'] / A,
    })
    
    # 计算总强度（所有分级通道）
    sum_all = scaled_df['ch1'] + scaled_df['ch2'] + scaled_df['ch4']
    log_sum = np.log10(sum_all + 1e-10)  # 避免log(0)
    
    # 构造观测向量 (N, 5, 1)
    N = len(scaled_df)
    obs_vector = np.zeros((N, 5, 1))
    obs_vector[:, 0, 0] = ratios['ch1/A'].values
    obs_vector[:, 1, 0] = ratios['ch2/A'].values
    obs_vector[:, 2, 0] = ratios['ch3/A'].values
    obs_vector[:, 3, 0] = ratios['ch4/A'].values
    obs_vector[:, 4, 0] = log_sum.values
    
    return obs_vector
```

### 步骤2：Codebook加载与构造

#### 2.1 加载PRISM30 Codebook

```python
def _load_codebook(self, codebook_path: Path) -> np.ndarray:
    """
    加载codebook并转换为PoSTcode格式。
    
    Codebook格式：CSV with columns ['Gene', 'ch1/A', 'ch2/A', 'ch3/A', 'ch4/A', 'log10_sum']
    转换为PoSTcode格式：(K, 5, 1) 其中K=31
    """
    codebook_df = pd.read_csv(codebook_path)
    
    # 提取比例和强度维度
    codebook_array = np.zeros((len(codebook_df), 5, 1))
    codebook_array[:, 0, 0] = codebook_df['ch1/A'].values
    codebook_array[:, 1, 0] = codebook_df['ch2/A'].values
    codebook_array[:, 2, 0] = codebook_df['ch3/A'].values
    codebook_array[:, 3, 0] = codebook_df['ch4/A'].values
    codebook_array[:, 4, 0] = codebook_df['log10_sum'].values
    
    return codebook_array, codebook_df['Gene'].tolist()
```

#### 2.2 构造背景Code（仅强度可学习）

```python
def _construct_background_code(self) -> np.ndarray:
    """
    构造背景code：前4维（比例）固定为0，第5维（强度）设为低值。
    
    返回：(1, 5, 1) 背景code
    """
    bg_code = np.zeros((1, 5, 1))
    bg_code[0, 4, 0] = -1.0  # 低强度值（log10空间）
    return bg_code
```

### 步骤3：修改PoSTcode模型（分通道缩放）

#### 3.1 修改`model_constrained_tensor`

需要修改`external/SPRINTseq-code/local/test/postcode/source-code/postcode/decoding_functions.py`中的`model_constrained_tensor`函数：

```python
# 原版（全局缩放）：
# codes_tr_v = pyro.param('codes_tr_v', 3 * torch.ones(1, D), constraint=constraints.greater_than(1.))

# 修改版（分通道缩放）：
codes_tr_v = pyro.param(
    'codes_tr_v', 
    3 * torch.ones(C),  # C = 通道数（5维）
    constraint=constraints.greater_than(1.)
)
```

#### 3.2 修改`theta`计算

```python
# 原版：
# theta = torch.matmul(codes * codes_tr_v + codes_tr_consts_v, mat_sqrt(sigma, D))

# 修改版（支持分通道缩放）：
# codes: (K, C, 1) -> reshape to (K, C)
# codes_tr_v: (C,)
# 使用广播机制进行element-wise乘法
codes_flat = codes.reshape(K, C)
theta = torch.matmul(
    (codes_flat * codes_tr_v.unsqueeze(0) + codes_tr_consts_v), 
    mat_sqrt(sigma, D)
)
```

**注意**：如果直接修改外部PoSTcode代码不方便，可以在`postcode_method.py`中创建一个包装函数，在调用PoSTcode之前/之后进行必要的转换。

### 步骤4：训练与推理

#### 4.1 调用PoSTcode训练

```python
def fit(self, data: pd.DataFrame, ...) -> "PostcodeMethod":
    """
    训练PoSTcode模型。
    """
    # 预处理
    processed_data = self.preprocess(data)
    obs_vector = self._construct_observation_vector(processed_data)
    
    # 加载codebook
    codebook, gene_names = self._load_codebook(self.codebook_path)
    
    # 添加背景code
    bg_code = self._construct_background_code()
    full_codebook = np.concatenate([codebook, bg_code], axis=0)
    
    # 调用PoSTcode训练
    from external.SPRINTseq_code.local.test.postcode.source_code.postcode.decoding_functions import decoding_function
    
    result = decoding_function(
        spots=obs_vector,  # (N, 5, 1)
        barcodes_01=full_codebook,  # (K+1, 5, 1)
        num_iter=self.config.get('num_iter', 60),
        batch_size=self.config.get('batch_size', 15000),
        estimate_bkg=True,  # 开启背景建模
        add_remaining_barcodes_prior=self.config.get('add_remaining_barcodes_prior', 0.05),
        print_training_progress=self.config.get('print_training_progress', True),
        # ... 其他参数
    )
    
    # 保存训练结果
    self.postcode_result = result
    self.gene_names = gene_names + ['Background']
    self.is_fitted = True
    
    return self
```

#### 4.2 推理与输出格式化

```python
def predict(self, data: pd.DataFrame) -> ClassificationResult:
    """
    预测并返回ClassificationResult。
    """
    # 预处理
    processed_data = self.preprocess(data)
    obs_vector = self._construct_observation_vector(processed_data)
    
    # 使用训练好的模型进行推理（E-step）
    class_probs = self._run_e_step(obs_vector)
    
    # 计算熵值
    entropy = self._calculate_entropy(class_probs)
    
    # 获取Top1基因
    top1_indices = np.argmax(class_probs, axis=1)
    top1_probs = class_probs[np.arange(len(class_probs)), top1_indices]
    top1_genes = [self.gene_names[i] if i < len(self.gene_names) else 'Infeasible' 
                  for i in top1_indices]
    
    # 构造ClassificationResult
    result = ClassificationResult(
        labels=top1_indices + 1,  # 1-based labels
        probabilities=class_probs,
        metadata={
            'method': 'Postcode',
            'gene_names': self.gene_names,
            'entropy': entropy,
            'top1_genes': top1_genes,
            'top1_probs': top1_probs,
        }
    )
    
    return result

def _calculate_entropy(self, class_probs: np.ndarray) -> np.ndarray:
    """
    计算香农熵：H = -Σ p_i * log(p_i)
    """
    probs_safe = np.clip(class_probs, 1e-10, 1.0)
    entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    return entropy
```

### 步骤5：输出格式（mapping.csv）

```python
def to_dataframe(self, result: ClassificationResult, intensity_df: pd.DataFrame) -> pd.DataFrame:
    """
    转换为DataFrame格式，包含所有类别的概率和熵值。
    
    输出列：
    - index: spot索引
    - Gene: Top1基因名
    - Probability: Top1概率
    - Entropy: 熵值
    - Prob_Gene_1, Prob_Gene_2, ..., Prob_Background, Prob_Infeasible: 所有类别的概率
    """
    # 基础列
    mapping_df = pd.DataFrame({
        'index': intensity_df['index'].values if 'index' in intensity_df.columns else intensity_df.index.values,
        'Gene': result.metadata['top1_genes'],
        'Probability': result.metadata['top1_probs'],
        'Entropy': result.metadata['entropy'],
    })
    
    # 添加所有类别的概率列
    gene_names = result.metadata['gene_names']
    for i, gene_name in enumerate(gene_names):
        mapping_df[f'Prob_{gene_name}'] = result.probabilities[:, i]
    
    # 如果有Infeasible类别
    if result.probabilities.shape[1] > len(gene_names):
        mapping_df['Prob_Infeasible'] = result.probabilities[:, len(gene_names):].sum(axis=1)
    
    return mapping_df
```

---

## 四、配置参数

### 4.1 方法配置（YAML）

```yaml
# configs/gene_calling_postcode.yaml
classification:
  method: "postcode"
  postcode:
    codebook_path: "configs/codebooks/prism30_codebook.csv"
    num_iter: 60
    batch_size: 15000
    estimate_bkg: true
    add_remaining_barcodes_prior: 0.05
    print_training_progress: true
    set_seed: 1
    # 分通道缩放相关
    use_channel_specific_scaling: true
    # 预处理相关
    p99_percentile: 99
    baseline_channel: "ch1"
```

### 4.2 数据规模优化

对于百万到千万级别的数据：
- **训练阶段**：使用全图数据，但可能需要分batch训练
- **推理阶段**：使用chunked inference（PoSTcode已支持）

---

## 五、测试计划

### 5.1 测试数据

- **路径**：`\\10.10.10.1\NAS Processed Images\20230523_HCC_PRISM_probe_refined_test_processed`
- **文件**：`intensity.csv`（包含position信息）

### 5.2 测试步骤

1. **Codebook生成验证**
   - 验证31种组合是否正确
   - 验证比例和是否为1

2. **预处理验证**
   - 验证P99缩放是否合理
   - 验证观测向量构造是否正确

3. **模型训练验证**
   - 验证分通道缩放是否生效
   - 验证背景建模是否仅作用于强度维度

4. **输出验证**
   - 验证概率矩阵维度是否正确（N × (31+1+1)）
   - 验证熵值计算是否正确
   - 验证mapping.csv格式是否符合要求

5. **与GMM方法对比**
   - 在同一数据集上运行两种方法
   - 对比基因判定的差异
   - 分析低强度点和ghost点的处理差异

---

## 六、潜在问题与解决方案

### 6.1 可识别性陷阱

**问题**：模型可能混淆"通道增益"和"基因身份"

**解决方案**：
- P99粗校正（已实现）
- 分通道缩放初始化接近1.0（因为已粗校正）

### 6.2 内存问题

**问题**：百万到千万级别数据可能导致OOM

**解决方案**：
- 使用chunked inference
- 训练时使用batch_size限制
- 考虑使用GPU加速（如果可用）

### 6.3 PoSTcode代码修改

**问题**：直接修改外部PoSTcode代码可能不方便

**解决方案**：
- 方案A：创建wrapper函数，在调用前后进行转换
- 方案B：fork PoSTcode代码到PRISM项目内，进行修改
- 方案C：使用monkey patching在运行时替换函数

---

## 七、实施优先级

1. **Phase 1（MVP）**：
   - ✅ Codebook生成工具
   - ⬜ 基础`PostcodeMethod`类（使用原始PoSTcode，不修改）
   - ⬜ 预处理流程（P99缩放 + 观测向量构造）
   - ⬜ 基础训练与推理

2. **Phase 2（增强）**：
   - ⬜ 分通道缩放实现
   - ⬜ 背景建模优化（仅强度可学习）
   - ⬜ 输出格式完善（所有类别概率 + 熵值）

3. **Phase 3（优化）**：
   - ⬜ 大规模数据优化
   - ⬜ 与GMM方法对比分析
   - ⬜ 性能调优

---

## 八、依赖项

- **PoSTcode相关**：
  - Pyro (>=1.0)
  - PyTorch
  - NumPy

- **PRISM现有依赖**：
  - pandas
  - scikit-learn（用于兼容性）

---

## 九、参考

- PoSTcode原始代码：`external/SPRINTseq-code/local/test/postcode/`
- SPRINTseq gene mapping实现：`external/SPRINTseq-code/local/test/gene_calling/gene_mapping.py`
- PRISM GMM方法：`code/src/gene_calling/methods/gmm_method.py`
