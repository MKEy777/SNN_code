import numpy as np
import matplotlib.pyplot as plt


def Renyi_spectral(A, Fs, p, window):
    """
    计算Renyi熵谱特征值。

    参数:
    A -- 输入矩阵（单导数据），一维时间序列
    Fs -- 采样率，单位为每秒采样点数
    p -- 绘图标志，若为1则绘制Renyi熵谱的曲线图
    window -- 窗口大小（秒），用于确定每次处理的信号长度

    返回:
    D_inf_all -- 每次计算得到的 q 趋于无穷大时的 Renyi 熵值序列
    D_q_0all -- 每次计算得到的 q=0 时的 Renyi 熵值序列
    D_q_1all -- 每次计算得到的 q=1 时的 Renyi 熵值序列
    average_D_inf -- D_inf_all 的平均值
    average_D_q_0 -- D_q_0all 的平均值
    average_D_q_1 -- D_q_1all 的平均值
    """
    # 输入信号的长度
    Len_signal = len(A)
    # 窗口时间（秒）
    window_t = window
    # 每次计算的序列长度（点数）
    M = int(Fs * window_t)
    # 每次滑动的点数
    slide_point = int(Fs * window_t)
    # 计算滑动窗口的次数
    h = int(np.floor((Len_signal - M) / slide_point))
    # 脑电测量设备的精准度
    delta_V = 0.01
    # 初始化存储列表
    D_inf_all = []
    D_q_0all = []
    D_q_1all = []
    M_D = []

    # 滑动窗口计算
    for j in range(h + 1):
        # 提取当前窗口的数据
        Xt = A[j * slide_point: j * slide_point + M]
        # 信号的最大和最小幅值
        X_max = np.max(Xt)
        X_min = np.min(Xt)
        # 划分的区间数
        N = int(np.floor((X_max - X_min) / delta_V))
        # 将信号映射到区间坐标
        Xt_axis = np.floor((Xt - X_min) / delta_V).astype(int) + 1
        # 获取唯一的区间值
        histograms = np.unique(Xt_axis)
        Point = len(histograms)
        # 初始化概率分布
        P_Xt = np.zeros(Point)
        for i, hist in enumerate(histograms):
            P_Xt[i] = np.sum(Xt_axis == hist)
        # 计算概率分布
        Pxt = P_Xt / M
        # 最大概率
        P_max = np.max(Pxt)
        # 当 q 趋于无穷大时的 Renyi 熵
        D_inf = np.log(P_max) / np.log(delta_V)
        # 存储结果
        D_inf_all.append(D_inf)

        # 定义 q 的取值范围
        q = np.arange(-50, 51, 1)
        q_num = len(q)
        D_q = np.zeros(q_num)
        for i, qi in enumerate(q):
            sum_pi_q = np.sum(Pxt ** qi)
            if qi != 1:
                D_q[i] = (1 / (qi - 1)) * (np.log(sum_pi_q) / np.log(delta_V))
            else:
                # 当 q=1 时，Renyi 熵退化为 Shannon 熵
                D_q[i] = -np.sum(Pxt * np.log(Pxt)) / np.log(1 / delta_V)
        # 提取 q=0 和 q=1 时的 Renyi 熵
        D_q_0 = D_q[50]  # q=0 对应索引50（-50 to 50）
        D_q_1 = D_q[51]  # q=1 对应索引51
        D_q_0all.append(D_q_0)
        D_q_1all.append(D_q_1)
        M_D.append(D_q)

        # 如果 p==1，绘制 q vs D_q 曲线
        if p == 1:
            plt.plot(q, D_q, 'k-', label=f'Window {j + 1}')

    # 计算平均值
    average_D_inf = np.mean(D_inf_all)
    average_D_q_0 = np.mean(D_q_0all)
    average_D_q_1 = np.mean(D_q_1all)
    # 计算 Renyi_entropy 的平均值（按列求均值）
    M_D = np.array(M_D).T  # 转置为 (q_num, h+1)
    Renyi_entropy = np.mean(M_D, axis=1)

    # 如果 p==1，绘制平均 Renyi_entropy 曲线并显示图形
    if p == 1:
        plt.plot(q, Renyi_entropy, 'r-', label='Average Renyi Entropy')
        plt.xlabel('q')
        plt.ylabel('D_q')
        plt.legend()
        plt.show()

    return D_inf_all, D_q_0all, D_q_1all, average_D_inf, average_D_q_0, average_D_q_1