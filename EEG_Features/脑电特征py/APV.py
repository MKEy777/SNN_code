def APV(data, x):
    """
    计算振幅变异系数(APV: Amplitude Parameter Variation)

    作者: zlx
    日期: 2022.04.25

    可以用bandpower提取各个频段的信息

    参数:
        data: 输入信号数据
        x: 参考信号

    返回:
        apv_val: APV值
    """

    # 初始化方差累加值
    Vr = 0

    # 获取数据长度
    N = len(data)

    # 计算信号的方差
    for i in range(N):
        Vr = data[i] ** 2 + Vr

    # 计算信号的平均方差
    Wi = Vr / N

    # 计算参考信号的均值
    Wo = np.mean(x)

    # 初始化总方差
    Wt = 0

    # 计算信号与参考信号的方差
    for i in range(1):
        Wt = (Wi - Wo) ** 2 + Wt

    # 计算delta值
    delt = Wt / 1

    # 计算并返回APV值
    apv_val = delt / Wo

    return apv_val