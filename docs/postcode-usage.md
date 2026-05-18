# PostcodeMethod 使用指南

## 快速开始

### 1. 在小数据集上测试

```bash
cd C:\Users\Mingchuan\Huanglab\spatial\PRISM\code
python scripts/gene_calling_postcode.py --run-id 20230523_HCC_PRISM_probe_refined_test_crop_processed --num-iter 30
```

### 2. 在大数据集上运行

```bash
python scripts/gene_calling_postcode.py --run-id 20230523_HCC_PRISM_probe_refined_test_processed --num-iter 60 --batch-size 20000
```

### 3. 启用分通道缩放

```bash
python scripts/gene_calling_postcode.py --run-id 20230523_HCC_PRISM_probe_refined_test_crop_processed --channel-specific
```

## 参数说明

- `--run-id`: 数据集运行ID（必需）
- `--channel-specific`: 启用分通道缩放（可选）
- `--num-iter`: 训练迭代次数（默认：60）
- `--batch-size`: 训练批次大小（默认：15000）

## 输出文件

运行完成后，会在 `\\10.10.10.1\NAS Processed Images\<run_id>\readout\mapping_postcode.csv` 生成结果文件。

输出列包括：
- `index`: spot索引
- `Gene`: Top1基因名
- `Probability`: Top1概率
- `Entropy`: 熵值
- `Prob_Gene_1`, `Prob_Gene_2`, ..., `Prob_Background`, `Prob_Infeasible`: 所有类别的概率

## 注意事项

1. **环境问题**：如果遇到numpy/pandas导入错误，请：
   - 关闭所有Python进程
   - 重新打开终端
   - 激活conda环境：`conda activate spatial-prep-dp`

2. **数据格式**：确保 `intensity.csv` 包含以下列之一：
   - 标准格式：`ch1`, `ch2`, `ch3`, `ch4`
   - Readout格式：`cy5`, `TxRed`, `cy3`, `FAM`（会自动转换）

3. **内存使用**：大数据集（百万级spots）可能需要：
   - 增加batch_size以减少内存占用
   - 使用GPU加速（如果可用）

## 故障排除

### 问题1：PoSTcode模块导入失败

**错误**：`ModuleNotFoundError: No module named 'postcode'`

**解决**：postcode 由 SPRINTseq 仓库 vendor，通过 editable install 进入 env：

```powershell
mamba run -n spatial-prep-dp pip install -e C:/Users/Mingchuan/Huanglab/SPRINTseq/experiments/src/postcode
```

确认安装：`mamba run -n spatial-prep-dp python -c "import postcode"`

### 问题2：Codebook文件未找到

**错误**：`FileNotFoundError: Codebook file not found`

**解决**：确保codebook文件存在：
```bash
# 应该存在：
# code/src/configs/codebooks/prism30_codebook.csv
```

### 问题3：通道名称不匹配

**错误**：`Missing required channels: ['ch1', 'ch2', ...]`

**解决**：检查intensity.csv的列名，脚本会自动转换readout格式（cy5, TxRed, cy3, FAM）到标准格式（ch1, ch2, ch3, ch4）
