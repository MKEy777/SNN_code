# 基于脉冲神经网络 (SNN) 的 EEG 情绪识别项目

## 概述

本项目旨在使用脉冲神经网络 (SNN) 对脑电图 (EEG) 信号进行情绪识别。项目实现了两种主要的 SNN 模型：基于速率编码的 Leaky Integrate-and-Fire (LIF) 模型和基于时间编码的首脉冲时间 (Time-To-First-Spike, TTFS) 模型。项目代码支持在公开的 DEAP 和 SEED 数据集上进行训练和评估。

代码库包含数据预处理、特征提取、模型定义、训练和评估等模块。

## 主要功能与特点

1.  **数据预处理**:
    * 支持 DEAP 和 SEED 数据集的处理。
    * 包含基线校正。
    * 提供带通滤波（`processing_*.py`）和不带通滤波（`processing_*_ORIGN.py`, `processing_*_NoBandpass*.py`）两种预处理选项。
    * 使用滑动窗口对数据进行分段。

2.  **特征提取与选择**:
    * 实现了 11 种 EEG 特征的提取，涵盖时频域和非线性动力学特征 (详见 `utils/extract_*.py` 和下面的特征列表)。这些特征主要用于 TTFS 模型。
    * 包含多种自动特征选择方法的描述 (基于原始 `README.md`)。

3.  **SNN 模型**:
    * **LIF 模型 (`model/LIF.py`, `train_lif.py`)**:
        * 基于速率编码。
        * 使用 Leaky Integrate-and-Fire 神经元。
        * 采用多种替代梯度函数 (Surrogate Gradient) 进行反向传播。
        * 直接使用分段后的 EEG 时序数据进行训练 (通过 `utils/load_dataset_deap.py` 加载)。
    * **TTFS 模型 (`model/TTFS_ORIGN.py`, `train_TTFS_*.py`)**:
        * 基于首脉冲时间编码 (Time-To-First-Spike)。
        * 使用 `SpikingDense` 层构建网络。
        * 包含动态时间参数调整机制。
        * 使用从 EEG 信号中提取的特征向量 (`.mat` 文件) 进行训练。

4.  **编码方式**:
    * **速率编码 (Rate Coding)**: 神经元的发放率代表信息 (LIF 模型)。
    * **时间编码 (Temporal Coding)**: 神经元的首次发放时间代表信息 (TTFS 模型)。

5.  **数据集支持**:
    * DEAP 数据集
    * SEED 数据集

## 项目结构

```
SNN_code/
├── model/                  # SNN 模型定义
│   ├── LIF.py              # LIF 神经元和层 (速率编码)
│   ├── TTFS.py             # (似乎是 TTFS 的另一版本或开发中版本)
│   └── TTFS_ORIGN.py       # TTFS 模型 (SpikingDense, 时间编码)
├── utils/                  # 数据处理和加载工具
│   ├── processing_deap.py      # DEAP 数据预处理 (带滤波, 分段) -> PerSession_MAT_Deap
│   ├── processing_deap_ORIGN.py# DEAP 数据预处理 (无滤波, 分段) -> PerSession_MAT_Deap_nofilter
│   ├── processing_seed.py      # SEED 数据预处理 (带滤波, 分段) -> PerSession_MAT
│   ├── processing_seed_ORIGN.py# SEED 数据预处理 (无滤波, 分段) -> PerSession_MAT_NoBandpass_Fixed
│   ├── extract_feature_11_deap.py  # DEAP 特征提取 (11种, 基于滤波数据) -> deap_features
│   ├── extract_feature_11_deap_NoBandpass.py # DEAP 特征提取 (11种, 基于无滤波数据) -> deap_features_NoFilter
│   ├── extract_features_11_seed.py # SEED 特征提取 (11种, 基于滤波数据) -> Individual_Features
│   ├── extract_features_11_seed_NoBanndpass.py # SEED 特征提取 (11种, 基于无滤波数据) -> Individual_Features_NoBandpass_Fixed
│   ├── load_dataset_deap.py    # DEAP 数据加载器 (用于 LIF 模型)
│   └── load_dataset_deap_ORIGN.py # (似乎是 load_dataset_deap 的另一版本)
├── train_lif.py            # 训练 LIF 模型的脚本 (使用 DEAP)
├── train_TTFS_deap.py      # 训练 TTFS 模型的脚本 (使用 DEAP 特征)
├── train_TTFS_deap_orign.py# 训练 TTFS 模型的脚本 (使用 DEAP 无滤波特征)
├── train_TTFS_seed.py      # 训练 TTFS 模型的脚本 (使用 SEED 特征)
├── train_TTFS_seed_orign.py# 训练 TTFS 模型的脚本 (使用 SEED 无滤波特征)
└── README.md               # (本文档)
```

## 依赖安装

需要安装以下 Python 库：

* PyTorch
* NumPy
* SciPy
* Scikit-learn
* Matplotlib
* tqdm

建议使用 `pip` 或 `conda` 进行安装。例如：

```bash
pip install torch numpy scipy scikit-learn matplotlib tqdm
```

## 使用方法

### 1. 数据准备

* **获取原始数据**: 下载 DEAP 和/或 SEED 数据集，并将其放置在项目代码能够访问的位置。例如，将 DEAP 的 `.dat` 文件放入 `./dataset` 目录，将 SEED 的预处理 EEG 数据放入 `./SEED/SEED/Preprocessed_EEG` 目录 (具体路径请根据实际情况调整)。
* **运行预处理脚本**:
    * **DEAP**:
        * 若需要带通滤波数据：运行 `python utils/processing_deap.py`。这会生成分段后的数据，保存在 `PerSession_MAT_Deap` 目录。
        * 若需要无滤波数据：运行 `python utils/processing_deap_ORIGN.py`。这会生成分段后的数据，保存在 `PerSession_MAT_Deap_nofilter` 目录。
    * **SEED**:
        * 若需要带通滤波数据：运行 `python utils/processing_seed.py`。这会生成分段后的数据，保存在 `PerSession_MAT` 目录。
        * 若需要无滤波数据：运行 `python utils/processing_seed_ORIGN.py`。这会生成分段后的数据，保存在 `PerSession_MAT_NoBandpass_Fixed` 目录。
* **运行特征提取脚本 (仅 TTFS 模型需要)**:
    * **DEAP**:
        * 基于滤波数据：运行 `python utils/extract_feature_11_deap.py` (输入目录为 `PerSession_MAT_Deap`)，特征保存到 `deap_features`。
        * 基于无滤波数据：运行 `python utils/extract_feature_11_deap_NoBandpass.py` (输入目录为 `PerSession_MAT_Deap_nofilter`)，特征保存到 `deap_features_NoFilter`。
    * **SEED**:
        * 基于滤波数据：运行 `python utils/extract_features_11_seed.py` (输入目录为 `PerSession_MAT`)，特征保存到 `Individual_Features`。
        * 基于无滤波数据：运行 `python utils/extract_features_11_seed_NoBanndpass.py` (输入目录为 `PerSession_MAT_NoBandpass_Fixed`)，特征保存到 `Individual_Features_NoBandpass_Fixed`。

**注意**: 请确保在运行脚本前，检查并修改脚本内部指定的 **输入** 和 **输出** 目录路径，使其与您的文件系统匹配。

### 2. 模型训练

* **训练 LIF 模型 (基于 DEAP 数据)**:
    1.  修改 `train_lif.py` 或 `utils/load_dataset_deap.py` 中的 `data_dir` 变量，指向 DEAP 原始 `.dat` 文件所在的目录 (例如 `./dataset`)。
    2.  运行训练脚本: `python train_lif.py`

* **训练 TTFS 模型**:
    1.  **修改特征目录**: 打开对应的 `train_TTFS_*.py` 脚本 (例如 `train_TTFS_deap.py` 或 `train_TTFS_seed_orign.py`)，找到 `FEATURE_DIR` 变量，将其修改为第 1 步中生成的特征 `.mat` 文件所在的目录 (例如 `./deap_features` 或 `./SEED/Individual_Features_NoBandpass_Fixed`)。
    2.  **运行训练脚本**: 例如，`python train_TTFS_deap.py` 或 `python train_TTFS_seed_orign.py`。

## 提取的特征列表 (供 TTFS 模型使用)

以下是 `utils/extract_*.py` 脚本提取的 11 种特征，这些特征会组合（例如，跨越不同频带）形成最终输入给 TTFS 模型的特征向量。

### 时频域特征 (Time-Frequency Domain Features)

1.  **峰峰值均值 (Peak-Peak Mean)**: 时间序列中峰（最大值）与谷（最小值）之间垂直长度的算术平均值。
    $$ \text{Peak-Peak Mean} = \frac{1}{N} \sum_{i=1}^{N} (x_{\text{max},i} - x_{\text{min},i}) $$
2.  **均方值 (Mean Square Value)**: 时间序列平方值的算术平均。
    $$ \text{Mean Square Value} = \frac{1}{T} \sum_{t=1}^{T} x(t)^2 $$
3.  **方差 (Variance)**: 反映时间序列的离散程度。
    $$ \text{Variance} = \frac{1}{T} \sum_{t=1}^{T} \left( x(t) - \mu \right)^2 $$
4.  **Hjorth 参数：活动性 (Activity)**: 等同于方差。
    $$ \text{Activity} = \text{Var}(x) $$
5.  **Hjorth 参数：移动性 (Mobility)**: 估计信号的平均频率。
    $$ \text{Mobility} = \sqrt{\frac{\text{Var}(x')}{\text{Var}(x)}} $$ ($x'$ 为信号一阶差分)
6.  **Hjorth 参数：复杂性 (Complexity)**: 反映信号频率变化的复杂度。
    $$ \text{Complexity} = \frac{\text{Mobility}(x')}{\text{Mobility}(x)} $$
7.  **最大功率谱频率 (Maximum Power Spectral Frequency)**: 功率谱密度最大值对应的频率。
8.  **最大功率谱密度 (Maximum Power Spectral Density)**: 功率谱中的最大值。
    $$ \text{Max PSD} = \max(P(f)) $$
9.  **功率总和 (Power Sum)**: 所有频率分量的功率之和。
    $$ \text{Power Sum} = \sum_{f} P(f) $$

### 非线性动态系统特征 (Non-linear Dynamical System Features) - 部分名称可能与代码实现略有差异

10. **香农熵 (Shannon Entropy)**: 衡量时间序列的不确定性。
    $$ H = -\sum_{i} p(x_i) \log_2 p(x_i) $$
11. **功率谱熵 (Power Spectral Entropy)**: 基于功率谱的熵，反映频谱复杂性。
    $$ \text{Power Spectral Entropy} = -\sum \tilde{P}(f) \log \tilde{P}(f) $$ ($\tilde{P}(f)$ 为归一化功率谱)

*(注：原始 `README.md` 中还包含其他非线性特征，如近似熵、C0复杂度等，上面的列表是基于 `extract_*.py` 文件中实际计算的特征)*

## 自动特征选择方法描述 (来自原始 README.md)

以下方法用于评估和选择最相关的特征，可能在项目研究中被使用或比较。

1.  **卡方检验 (Chi-Squared Test, χ²)**: 评估离散特征与目标变量的独立性。
    $$ \chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{i,j} - E_{i,j})^2}{E_{i,j}} $$
2.  **互信息 (Mutual Information, MI)**: 量化特征与目标变量之间的信息共享程度（线性与非线性）。
    $$ I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)} $$
3.  **ANOVA F 值 (ANOVA F-Value)**: 通过方差分析衡量连续特征对分类目标的区分能力。
    $$ F_{\text{ratio}} = \frac{\sigma_{\text{between}}^2}{\sigma_{\text{within}}^2} $$
4.  **递归特征消除 (Recursive Feature Elimination, RFE)**: 基于分类器权重递归剔除最不重要特征。
5.  **L1 正则化 (L1-Norm Penalty)**: 在分类器目标函数中加入 L1 惩罚项，诱导稀疏权重，实现嵌入式特征选择。
    $$ \min_{\omega_0, \omega} \sum_{i=1}^n \left[1 - y_i(\omega_0 + \omega^T x_i)\right]_+ + C \|\omega\|_1 $$

## 数据集

* **DEAP**: [http://www.eecs.qmul.ac.uk/mmv/datasets/deap/](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
* **SEED**: [https://bcmi.sjtu.edu.cn/home/seed/seed.html](https://bcmi.sjtu.edu.cn/home/seed/seed.html)

请自行下载数据集并按需放置。

## 注意事项

* 运行代码前，请务必检查并修改脚本中涉及的 **文件路径** (数据目录、输出目录、特征目录等)，使其指向您本地的正确位置。
* 不同的训练脚本对应不同的模型 (LIF/TTFS)、数据集 (DEAP/SEED) 和预处理/特征版本 (带滤波/无滤波)。请根据您的需求选择合适的脚本运行。
```