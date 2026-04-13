# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本项目中工作时提供指导。

## 项目概述

这是一个基于 TSLib（清华大学 THUML 实验室的时间序列库）的深度学习时间序列预测项目，核心模型为 **Hankelformer**——一种结合 Hankel 矩阵变换和对比学习的长时预测模型。

## 常用命令

### 运行实验
```bash
# 使用 Hankelformer 进行长时预测
python run.py --task_name long_term_forecast_contra --is_training 1 --model Hankelformer --data <数据集> --seq_len 96 --pred_len 96 --learning_rate 0.0005 --window_size <窗口大小> --contrastive_weight <对比权重>

# 测试模式（设置 is_training 0）
python run.py --task_name long_term_forecast_contra --is_training 0 --model Hankelformer --data <数据集> --seq_len 96 --pred_len 96
```

### 示例脚本
```bash
# ETT 数据集
bash scripts/long_term_forecast/ETT_script/Hankelformer_ETTh1.sh

# 电力数据集
bash scripts/long_term_forecast/ECL_script/Hankelformer.sh
```

### 关键参数
- `--task_name`: 任务类型，`long_term_forecast` 或 `long_term_forecast_contra`
- `--model`: 模型名称（Hankelformer、Autoformer、TimesNet 等）
- `--is_training`: 1 表示训练，0 表示测试
- `--window_size`: Hankel MLP 窗口大小（默认 1）
- `--contrastive_weight`: 对比学习损失权重（默认 0.05）
- `--seq_len`: 输入序列长度
- `--pred_len`: 预测序列长度
- `--learning_rate`: 学习率（电力数据集建议使用 0.0005）

## 架构

### 目录结构
- `models/`: 神经网络模型（包括 Hankelformer 各变体、Autoformer、Transformer 等）
- `exp/`: 实验类，封装训练/测试流程
- `layers/`: 可复用组件（编码器、注意力机制、嵌入层等）
- `data_provider/`: 数据加载与预处理
- `utils/`: 评估指标、工具函数、时间特征
- `scripts/`: 运行基准测试的 Shell 脚本

### Hankelformer 模型
位于 [models/Hankelformer.py](models/Hankelformer.py)：
- **HankelMLP**: 将输入序列转换为 Hankel 矩阵形式并用 MLP 处理
- **Transformer Encoder**: 标准 Transformer 编码器层
- **融合层**: 将原始表示和 Hankel 处理后的表示进行融合
- **对比学习**: 使用 InfoNCE 损失进行表征学习

主要模型变体：
- `Hankelformer.py` - 完整版本，包含 Hankel 变换 + 对比学习
- `Hankelformer_base.py` - 基线版本
- `Hankelformer_without_hankel.py` - 移除 Hankel 变换
- `Hankelformer_without_Contrastive.py` - 移除对比学习

### 实验流程
1. [exp/exp_basic.py](exp/exp_basic.py): 基类，包含模型注册表
2. [exp/exp_long_term_forecasting.py](exp/exp_long_term_forecasting.py): 标准长时预测
3. [exp/exp_long_term_forecasting_contrastive.py](exp/exp_long_term_forecasting_contrastive.py): Hankelformer 专用，包含对比学习损失

### 数据格式
数据集位于 `dataset/` 目录，使用 CSV 格式。通过 `--root_path` 和 `--data_path` 参数指定。

## 依赖
```
torch, numpy, pandas, scikit-learn, matplotlib, tqdm, einops, PyWavelets
```
完整依赖见 [requirements.txt](requirements.txt)。

## 注意事项
- 该库使用统一接口支持多种时间序列任务（预测、缺失值填充、分类、异常检测）
- 模型注册表位于 `exp/exp_basic.py`，将模型名称映射到对应类
- 默认为 `./checkpoints/` 目录保存模型检查点
- 随机种子固定为 2023，确保可复现性