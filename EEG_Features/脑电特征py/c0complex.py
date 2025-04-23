import numpy as np
import matplotlib.pyplot as plt

def c0complex(A, Fs, p, window):
    """
    计算C0复杂度特征值。

    参数:
    A -- 输入矩阵（单导数据）
    Fs -- 采样率
    p -- 如果等于1则画出C0的曲线图
    window -- 窗口大小（秒）

    返回:
    C0 -- C0特征值序列
    C0_average -- C0的平均值
    """
    # 获取输入数据的长度
    M = len(A)
    # 窗口时间（秒）
    window_t = window
    # 每次计算的序列长度（点数）
    N = int(Fs * window_t)
    # 每次滑动的点数
    m = int(Fs * window_t)
    # 阈值常数
    r = 5
    # 计算可以滑动的次数
    t = (M - N) / m
    # 取整，得到滑动的总次数
    h = int(np.floor(t))
    # 初始化C0序列
    C0 = []

    # 滑动窗口计算C0复杂度
    for i in range(h + 1):
        # 数据滑动读取，从A中提取当前窗口的数据
        data = A[i * m : i * m + N]
        # 对当前窗口数据进行快速傅里叶变换（FFT）
        Fn = np.fft.fft(data, N)
        # 初始化规则部分的频谱
        Fn_1 = np.zeros_like(Fn)
        # 计算频谱的平方和
        Gsum = 0
        for j in range(N):
            Gsum += np.abs(Fn[j]) ** 2
        # 计算频谱的均方值
        Gave = (1 / N) * Gsum
        # 根据阈值提取规则部分的频谱
        for j in range(N):
            if np.abs(Fn[j]) ** 2 > (r * Gave):
                Fn_1[j] = Fn[j]
        # 通过逆FFT得到规则部分的数据
        data1 = np.fft.ifft(Fn_1, N)
        # 计算随机部分的平方
        D = (np.abs(data - data1)) ** 2
        # 计算随机部分的面积
        Cu = np.sum(D)
        # 计算原始数据的平方
        E = (np.abs(data)) ** 2
        # 计算原始数据的面积
        Cx = np.sum(E)
        # 计算并存储C0复杂度
        C0.append(Cu / Cx)

    # 如果p等于1，则绘制C0的曲线图
    if p == 1:
        plt.plot(C0)
        plt.title("C0 Complexity Curve")
        plt.xlabel("Window Index")
        plt.ylabel("C0 Value")
        plt.show()

    # 计算C0的平均值
    C0_average = np.mean(C0)

    # 返回C0序列和平均值
    return C0, C0_average