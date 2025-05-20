#基于脉冲神经网络 (SNN) 的 EEG 情绪识别

## 简介

本项目旨在使用脉冲神经网络 (SNN) 对脑电图 (EEG) 信号进行处理和分析，以实现情绪识别。项目主要使用了 SEED 数据集进行模型的训练和评估。代码包含了数据预处理、特征提取、SNN 模型构建以及模型训练等模块。

## 项目特点

  * **EEG 数据处理**: 针对 SEED 数据集定制了多种预处理流程，包括基线校正等。
  * **特征提取**: 从处理后的 EEG 信号中提取有效特征，支持不使用带通滤波的特征提取方式。
  * **脉冲神经网络模型**: 实现了一个基于首次脉冲时间 (TTFS) 编码的 SNN 模型。
  * **模型训练**: 提供了原始模型训练脚本以及可能包含 L1/L2 正则化等改进策略的训练脚本。

## 项目结构

```
.
├── model/
│   └── TTFS_ORIGN.py             # TTFS 脉冲神经网络模型定义
├── utils/
│   ├── processing_seed_all_baselinecorrected.py  # SEED 数据集预处理 (带基线校正)
│   ├── processing_seed_ORIGN.py                # SEED 数据集原始预处理流程
│   └── extract_features_11_seed_NoBanndpass.py # SEED 数据集特征提取 (不使用带通滤波)
├── train_orign.py                # 原始模型的训练脚本
├── train_randomforest.py                # 包含随机森林特征提取的模型训练脚本
└── README.md                     # 本文档
```

## 环境依赖

本项目主要使用 Python 语言编写，建议在以下环境中运行：

  * Python 3.x
  * NumPy
  * SciPy
  * Scikit-learn
  * Pytorch 

具体的依赖库及其版本请参照代码文件中的导入语句。建议使用 `pip` 或 `conda` 创建独立的虚拟环境来管理项目依赖。

```bash
pip install numpy scipy scikit-learn torch mne
```

## 使用说明

### 1\. 数据准备

  * **获取 SEED 数据集**: 请自行从官方渠道下载 SEED 数据集。
  * **数据预处理**:
      * 使用 `utils/processing_seed_ORIGN.py` 或 `utils/processing_seed_all_baselinecorrected.py` 脚本对原始 SEED 数据集进行预处理。
    <!-- end list -->
    ```bash
    python utils/processing_seed_ORIGN.py <相关参数>
    # 或者
    python utils/processing_seed_all_baselinecorrected.py <相关参数>
    ```
    请根据脚本内部的说明或注释修改输入输出路径及相关参数。
  * **特征提取**:
      * 使用 `utils/extract_features_11_seed_NoBanndpass.py` 脚本从预处理后的数据中提取特征。
    <!-- end list -->
    ```bash
    python utils/extract_features_11_seed_NoBanndpass.py <相关参数>
    ```
    同样，请根据脚本内部的说明或注释修改输入输出路径及相关参数。

### 2\. 模型训练

预处理和特征提取完成后，可以使用提供的训练脚本来训练 SNN 模型：

  * **训练原始模型**:
    ```bash
    python train_orign.py <相关参数>
    ```
  * **训练带 L1/L2 正则化的模型 (如果适用)**:
    ```bash
    python train_L1_L2.py <相关参数>
    ```

脚本中的参数可能包括学习率、训练轮数、批处理大小、数据路径等。请查看对应脚本文件获取详细的参数配置信息。训练好的模型通常会保存在指定的路径下。

## 模型介绍

本项目采用的脉冲神经网络模型是基于首次脉冲时间 (Time to First Spike, TTFS) 编码的。TTFS 是一种常见的脉冲编码方式，神经元的输出信息由其首次发放脉冲的时间来表示。具体的网络结构和参数配置请参考 `model/TTFS_ORIGN.py` 文件。

## 数据集

本项目主要针对 [SEED (SJTU Emotion EEG Dataset)](http://bcmi.sjtu.edu.cn/home/seed/) 数据集进行开发和测试。该数据集包含了多名被试在观看不同情绪诱导电影片段时的 EEG 信号，广泛应用于情绪识别研究。
