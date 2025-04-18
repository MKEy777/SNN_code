import numpy as np
import scipy.io as sio
import os

# 1. 加载 EEG 数据
# 从指定路径读取 MATLAB 格式的 EEG 预处理数据（针刺前阶段）
EEG_pre = sio.loadmat(r'E:\1科研\代码\脑电分析代码\Preprocessing\Acupuncture\sub_eeg\sub16_pre_split.mat')

# 2. 参数设置
ntrial = 15        # 每个条件（除针刺中期）对应的试次数目
con_sub = 3        # 条件数量：0-针刺前，1-针刺中，2-针刺后
nchannel = 3       # 通道数量（示例值）
fs = 250           # 采样率（Hz）
n_point = 2000     # 每段 EEG 数据的点数

# 3. 特征存储字典初始化
feat_sub = {}       # 单被试原始特征集合
feat_sub_mean = {}  # 单被试按 trial 均值后的特征集合

# =============================================================================
# 以下部分为示例的单被试特征提取逻辑，已用中文注释说明
# =============================================================================
# for sub in range(con_sub):
#     # 针刺中期（sub==1）试验数为225，其余阶段为15
#     ntrial = 225 if sub == 1 else 15
#     # con_set: 存储每个 trial 的所有时段特征（行数 = ntrial * 3），8个特征列
#     con_set = np.zeros((ntrial * 3, 8))
#     # con_set_mean: 存储每个 trial 的特征均值（行数 = ntrial），8个特征列
#     con_set_mean = np.zeros((ntrial, 8))
#
#     for trial in range(ntrial):
#         # 提取当前阶段、当前 trial 的 EEG 数据
#         EEG = EEG_pre['sub16_pre_split'][sub][trial]
#         # 调用自定义特征提取函数 tiqutz12，输出形状假设为 (3, 8)
#         tmp = tiqutz12(EEG, EEG)
#         # 将提取结果按行填入 con_set
#         con_set[(trial*3):(trial*3+3), :] = tmp
#         # 计算该 trial 上的 8 维特征均值
#         con_set_mean[trial, :] = np.mean(tmp, axis=0)
#
#     # 保存当前阶段的所有时段特征矩阵
#     feat_sub[sub] = con_set
#
#     # 针刺中期数据再按每15次 trial 求一次均值，保持与 pre/post 一致的 15 行
#     if sub == 1:
#         con_set_mean_tmp = np.zeros((15, 8))
#         for i in range(15):
#             start = i * 15
#             con_set_mean_tmp[i, :] = np.mean(con_set_mean[start:start + 15, :], axis=0)
#         feat_sub_mean[sub] = con_set_mean_tmp
#     else:
#         # 其他阶段直接使用每 trial 的均值
#         feat_sub_mean[sub] = con_set_mean
#
# # 将单被试特征保存为 .mat 文件
# sio.savemat(r'E:\1科研\代码\脑电分析代码\Preprocessing\Acupuncture\sub_eeg\sub16_split_feat.mat',
#             {'feat_sub': feat_sub})
# sio.savemat(r'E:\1科研\代码\脑电分析代码\Preprocessing\Acupuncture\sub_eeg\sub16_split_feat_mean.mat',
#             {'feat_sub_mean': feat_sub_mean})

# =============================================================================
# 4. 加载所有被试的特征文件（示例）
# =============================================================================
# data_sub = {}
# for i in range(16):
#     path = fr'E:\1科研\代码\脑电分析代码\Preprocessing\Acupuncture\sub_eeg\sub{i}_split_feat.mat'
#     data_sub[i] = sio.loadmat(path)

# =============================================================================
# 5. 初始化跨被试平均所需数组
# =============================================================================
# 针刺前：15 段 × 8 特征
data_sub_mean_pre = np.zeros((15, 8))
# 针刺中：225 段 × 8 特征
data_sub_mean_acu = np.zeros((225, 8))
# 针刺后：15 段 × 8 特征
data_sub_mean_post = np.zeros((15, 8))

# 临时矩阵：行对应时段点数，列对应被试编号，用于汇总
# pre/post 每个阶段段数为15，被试数为15
data_tmp_pre = np.zeros((15, 15))
data_tmp_post = np.zeros((15, 15))
# 针刺中期段数为225，被试数为15
data_tmp_acu = np.zeros((225, 15))

# 用于存储跨被试各条件的平均特征
acu_feat_sub_mean_delta = {}

# =============================================================================
# 6. 计算跨被试平均特征
# =============================================================================
for k in range(3):
    # k=0: pre, k=1: acu, k=2: post
    for j in range(8):  # 遍历8个特征维度
        for i in range(15):  # 遍历15个被试
            # 从每个被试的数据字典中提取对应阶段的特征矩阵
            data_tmp_pre[:, i] = data_sub[i]['feat_sub'][0, 0][:, j]   # 针刺前
            data_tmp_acu[:, i] = data_sub[i]['feat_sub'][0, 1][:, j]   # 针刺中
            data_tmp_post[:, i] = data_sub[i]['feat_sub'][0, 2][:, j]  # 针刺后

        # 对所有被试在每个时间段上求均值，得到平均特征曲线
        data_sub_mean_pre[:, j] = np.mean(data_tmp_pre, axis=1)
        data_sub_mean_acu[:, j] = np.mean(data_tmp_acu, axis=1)
        data_sub_mean_post[:, j] = np.mean(data_tmp_post, axis=1)

    # 将当前条件的跨被试平均结果保存到字典
    acu_feat_sub_mean_delta[k] = {
        'pre': data_sub_mean_pre,
        'acu': data_sub_mean_acu,
        'post': data_sub_mean_post
    }

# =============================================================================
# 7. 抑郁组数据处理框架（示例，需定义 tiqutz128）
# =============================================================================
# feat_dep = {}
# for sub in range(dep_sub):
#     # 初始化抑郁组每 trial 特征矩阵，通道数*5 为示例
#     dep_set = np.zeros((ntrial, nchannel * 5))
#     for trial in range(ntrial):
#         EEG = depression[sub][trial]
#         # 使用另一种特征提取函数 tiqutz128
#         dep_set[trial, :] = tiqutz128(EEG)
#     # 将提取结果与其他相关变量 a 拼接保存
#     feat_dep[sub] = np.column_stack([a, dep_set])
