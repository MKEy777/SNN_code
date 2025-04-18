import numpy as np
from scipy import signal


def bandpassx(x, fs, fc1, fc2):
    """
    基本带通FIR滤波器设计函数

    功能: 使用汉宁窗设计带通FIR滤波器,具有较小的旁瓣和较大的衰减速度,可以实时处理

    参数:
        x: 输入信号
        fs: 采样频率
        fc1: 滤波器下限频率
        fc2: 滤波器上限频率

    返回:
        output: 滤波后的信号(与输入信号长度保持一致,前端数据基本无失真)

    注意:
        此程序在2010.12.30由Yanbing Qi开发,版本0.1
    """

    # 检查输入信号维度
    x = np.array(x)
    if len(x.shape) > 1:
        a_1, b_1 = x.shape
    else:
        a_1, b_1 = x.shape[0], 1

    updown = 0  # 标记是否需要转置

    # 如果输入是列向量,转为行向量
    if b_1 == 1 and b_1 < a_1:
        x = x.T
        updown = 1

    # 更新维度信息
    if len(x.shape) > 1:
        a_1, b_1 = x.shape
    else:
        a_1, b_1 = x.shape[0], 1

    len_x = len(x)

    # 计算归一化频率
    wp = [2 * fc1 / fs, 2 * fc2 / fs]
    N = 512  # 滤波器阶数

    # 输入检查
    if (a_1 != 1 and b_1 != 1):
        raise ValueError('输入矩阵不是一维数组,请检查输入')

    if (fc1 < 0.2 or fc2 > 512):
        raise ValueError('输入低频必须大于0.2Hz,高频必须小于512Hz')

    # 设计FIR带通滤波器
    b = signal.firwin(N + 1, wp, window='hann', pass_zero=False)

    # 进行滤波
    output1 = np.convolve(b, x)

    # 截取与输入信号等长的输出
    M = (N - 1) // 2
    output = output1[M:M + len_x]

    # 如果输入是列向量,输出也转为列向量
    if updown == 1:
        output = output.T

    return output