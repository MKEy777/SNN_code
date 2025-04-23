import numpy as np

# 定义全局参数
ntrial = 8          # 试验次数
con_sub = 108       # 正常被试（对照组）的数量
# dep_sub = 24       # 抑郁被试（训练组）的数量（注释掉的部分）
nchannel = 3        # 通道数量
fs = 250            # 采样频率（Hz）
n_point = 500       # 脑电数据点数

# 假设 EEG_Normal_Data 是一个列表，包含每个正常被试的脑电数据
# EEG_Normal_Data[sub][trial] 代表第 sub 个被试的第 trial 次试验的脑电数据

# 存储正常被试的特征
feat_Normal_0401 = [None] * con_sub

# 处理正常被试数据
for sub in range(con_sub):
    # 初始化特征矩阵，形状为 (ntrial, 7)
    con_set = np.zeros((ntrial, 7))
    for trial in range(ntrial):
        # 获取第 sub 个被试的第 trial 次试验的脑电数据
        EEG = EEG_Normal_Data[sub][trial]
        # 调用 tiqutz12 函数提取特征，假设返回一个包含7个特征的数组
        con_set[trial, :] = tiqutz12(EEG)
    # 将提取的特征存储到 feat_Normal_0401 中
    feat_Normal_0401[sub] = con_set

# 以下是抑郁被试部分的转换（原MATLAB代码中被注释掉）
# 如果需要使用，请取消注释并确保相关数据和函数可用
"""
# 存储抑郁被试的特征
feat_dep = [None] * dep_sub

# 处理抑郁被试数据
for sub in range(dep_sub):
    # 初始化特征矩阵，形状为 (ntrial, nchannel * 5)
    dep_set = np.zeros((ntrial, nchannel * 5))
    # 创建抑郁标签数组，形状为 (ntrial, 1)，全为1
    a = np.ones((ntrial, 1))
    for trial in range(ntrial):
        # 获取第 sub 个被试的第 trial 次试验的脑电数据
        EEG = depression[sub][trial]
        # 调用 tiqutz128 函数提取特征，假设返回一个包含 nchannel*5 个特征的数组
        dep_set[trial, :] = tiqutz128(EEG)
    # 将标签和特征拼接后存储到 feat_dep 中
    feat_dep[sub] = np.hstack((a, dep_set))
"""