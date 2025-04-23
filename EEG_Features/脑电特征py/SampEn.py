import numpy as np

def SampEn(dim, r, data, tau=1):
    """
    计算给定时间序列数据的样本熵（Sample Entropy）。

    参数:
    dim : int
        嵌入维度。
    r : float
        容差（通常为0.2 * 标准差）。
    data : array_like
        时间序列数据。
    tau : int, 可选
        降采样延迟时间（默认为1）。

    返回:
    saen : float
        计算得到的样本熵值。
    """
    if tau > 1:
        data = data[::tau]  # 降采样

    N = len(data)
    correl = np.zeros(2)
    dataMat = np.zeros((dim + 1, N - dim))

    for i in range(dim + 1):
        dataMat[i, :] = data[i:N - dim + i]

    for m in range(dim, dim + 2):
        count = np.zeros(N - dim)
        tempMat = dataMat[:m, :]

        for i in range(N - dim):
            # 计算切比雪夫距离，排除自我匹配
            dist = np.max(np.abs(tempMat[:, i + 1:N - dim + 1] - tempMat[:, i][:, np.newaxis]), axis=0)
            # 计算距离的Heaviside函数
            D = (dist < r)
            count[i] = np.sum(D) / (N - dim)

        correl[m - dim] = np.sum(count) / (N - dim)

    saen = np.log(correl[0] / correl[1]) if correl[1] != 0 else float('inf')
    return saen