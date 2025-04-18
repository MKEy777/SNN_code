import numpy as np


def LZC(data):
    """
    计算数据的Lempel-Ziv Complexity（LZC）。

    参数:
    data : numpy.ndarray
        输入数据矩阵，可以是多列的。

    返回:
    lzc : float
        计算得到的LZC值。
    """
    # 数据二值化处理
    # 计算每列的中位数
    median_data = np.median(data, axis=0)

    # 获取数据的行数和列数
    l, c = data.shape

    # 初始化二值化数据矩阵
    binary_data = np.zeros_like(data, dtype=str)
    binary_data.fill('0')

    # 对每列进行二值化
    for i in range(c):
        # 找到大于中位数的元素
        Tno = data[:, i] > median_data[i]
        # 将对应位置设为'1'
        binary_data[Tno, i] = '1'

    # 将二值化数据转换为一维字符串
    # 假设将所有列连接成一个长字符串进行LZC计算
    x = ''.join(binary_data.flatten())

    # 计算LZC
    c = 1  # 模式初始值
    S = x[0]  # S初始化为第一个元素
    Q = ''  # Q初始化为空
    SQ = ''  # SQ初始化为空

    for i in range(1, len(x)):
        Q += x[i]  # 将当前元素追加到Q
        SQ = S + Q  # 将S和Q连接成SQ
        SQv = SQ[:-1]  # SQv是SQ去掉最后一个元素
        if Q not in SQv:  # 如果Q不是SQv的子串
            S = SQ  # 更新S为SQ
            Q = ''  # 清空Q
            c += 1  # 模式计数器加1

    # 计算b和lzc
    b = len(x) / np.log2(len(x)) if len(x) > 0 else 0
    lzc = c / b if b > 0 else 0

    return lzc