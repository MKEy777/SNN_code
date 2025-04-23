import numpy as np
from scipy.signal import spectrogram, hamming


def FAA(EEG):
    """
    计算脑电数据的FAA值。

    参数:
    EEG : numpy.ndarray
        脑电数据矩阵，包含多个通道的数据。

    返回:
    FAA : float
        计算得到的FAA值。
    """
    # 提取左通道和右通道数据
    # MATLAB中索引从1开始，Python中从0开始，因此列1变为0，列3变为2
    L = EEG[:, 0]  # 左通道数据
    R = EEG[:, 2]  # 右通道数据

    # 定义频率范围（8-13 Hz）
    FREQ_1 = 8
    FREQ_2 = 13

    # 计算左通道的功率谱
    window_length_L = int(np.floor(len(L) / 2))  # 计算窗口长度，取信号长度的一半
    WIND_L = hamming(window_length_L)  # 使用Hamming窗
    OVER_L = int(np.floor(len(L) / 1.5 / 2))  # 设置50%重叠
    SIGN_L = L  # 获取信号，Python中spectrogram接受1D数组，无需转置
    f_L, t_L, Sxx_L = spectrogram(SIGN_L, fs=250, window=WIND_L, noverlap=OVER_L)  # 计算spectrogram
    indFreqs_L = np.where((f_L > FREQ_1) & (f_L < FREQ_2))[0]  # 找到8-13 Hz之间的频率索引
    POW_L = Sxx_L[indFreqs_L, :]  # 提取对应频率的功率谱

    # 计算右通道的功率谱
    window_length_R = int(np.floor(len(R) / 2))  # 计算窗口长度，取信号长度的一半
    WIND_R = hamming(window_length_R)  # 使用Hamming窗
    OVER_R = int(np.floor(len(R) / 1.5 / 2))  # 设置50%重叠
    SIGN_R = R  # 获取信号
    f_R, t_R, Sxx_R = spectrogram(SIGN_R, fs=250, window=WIND_R, noverlap=OVER_R)  # 计算spectrogram
    indFreqs_R = np.where((f_R > FREQ_1) & (f_R < FREQ_2))[0]  # 找到8-13 Hz之间的频率索引
    POW_R = Sxx_R[indFreqs_R, :]  # 提取对应频率的功率谱

    # 计算FAA值
    # Python中spectrogram返回实数功率谱密度，无需abs()
    FAA = np.mean(np.log(POW_R) - np.log(POW_L))  # 计算左右通道功率谱对数差的平均值

    return FAA