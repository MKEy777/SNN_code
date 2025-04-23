import numpy as np

def reconstitution(data, N, m, tau):
    """
    重构相空间。

    参数:
        data -- 输入时间序列（一维数组或列表）
        N -- 时间序列的长度，即 len(data)
        m -- 嵌入空间维数
        tau -- 时间延迟
    返回:
        X -- 重构的相空间矩阵，形状为 (m, M)，其中 M = N - (m - 1) * tau
    """
    # 计算相空间中点的个数 M
    M = N - (m - 1) * tau
    # 初始化相空间矩阵 X，形状为 (m, M)
    X = np.zeros((m, M))
    # 相空间重构
    for j in range(M):
        for i in range(m):
            # 计算 data 中的索引：i * tau + j
            # Python 索引从 0 开始，X[i, j] 对应 data[i * tau + j]
            X[i, j] = data[i * tau + j]
    return X