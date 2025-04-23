import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def reC(A, Fs, p, window):
    """
    计算关联维数D2特征值。

    参数:
    A -- 输入矩阵（单导数据）
    Fs -- 采样率
    p -- 如果等于1则画出相关曲线和D2曲线图
    window -- 窗口大小（秒）

    返回:
    CorrelationDimension -- D2特征值序列
    M_C -- D2的平均值
    """
    # 窗口时间（秒）
    window_t = window
    # 每次计算的序列长度（点数）
    N = int(Fs * window_t)
    # 输入数据的总长度
    G = len(A)
    # 每次滑动的点数
    g = int(Fs * window_t)
    # 计算可以滑动的次数
    t = (G - N) / g
    h = int(np.floor(t))
    # 相关积分的r值数量
    ss = 20
    # 嵌入维数
    m = 15
    # 初始化D2序列
    CorrelationDimension = []

    # 滑动窗口计算
    for ii in range(h + 1):
        # 数据滑动读取
        data = A[ii * g : ii * g + N]
        # 估计延迟时间tau（需要定义tau_def函数）
        tau = tau_def(data)
        # 相空间中每维序列的长度
        M = N - (m - 1) * tau
        # 重构相空间（需要定义reconstitution函数）
        Y = reconstitution(data, N, m, tau)
        # 初始化距离矩阵
        d = np.zeros((M - 1, M))
        # 计算状态空间中每两点之间的距离
        for i in range(M - 1):
            for j in range(i + 1, M):
                d[i, j] = np.linalg.norm(Y[:, i] - Y[:, j], 2)
        # 找到最大距离
        max_d = np.max(d)
        # 找到最小非零距离
        min_d = max_d
        for i in range(M - 1):
            for j in range(i + 1, M):
                if d[i, j] != 0 and d[i, j] < min_d:
                    min_d = d[i, j]
        # r的步长
        delt = (max_d - min_d) / ss
        # 初始化r和C
        r = np.zeros(ss)
        C = np.zeros(ss)
        ln_C = np.zeros(ss)
        ln_r = np.zeros(ss)
        # 计算关联积分
        for k in range(ss):
            r[k] = min_d + (k + 1) * delt  # 从min_d开始递增
            # 统计小于r的距离数量
            H = np.sum(d < r[k])
            C[k] = 2 * H / (M * (M - 1)) - 1
            ln_C[k] = np.log2(C[k])
            ln_r[k] = np.log2(r[k])
        # 如果p==1，绘制log(r) vs ln_C的曲线
        if p == 1:
            plt.subplot(1, 2, 1)
            plt.plot(ln_r, ln_C, '+--')
            plt.grid(True)
            plt.xlabel('log(r)')
            plt.ylabel('ln_C(m,:)')
        # 拟合线性区域，选择索引[1:5]对应MATLAB的[2:6]
        LinearZone = slice(1, 5)
        # 使用线性回归拟合获取斜率（D2）
        slope, intercept, _, _, _ = linregress(ln_r[LinearZone], ln_C[LinearZone])
        CorrelationDimension.append(slope)

    # 计算D2的平均值
    M_C = np.mean(CorrelationDimension)
    # 如果p==1，绘制D2序列曲线
    if p == 1:
        plt.subplot(1, 2, 2)
        plt.plot(CorrelationDimension)
        plt.ylabel('CorrelationDimension')
        plt.show()

    return CorrelationDimension, M_C

# 注意：以下两个函数需要在代码中额外定义
def tau_def(data):
    """估计延迟时间tau的函数（需自行实现）"""
    pass

def reconstitution(data, N, m, tau):
    """重构相空间的函数（需自行实现）"""
    pass