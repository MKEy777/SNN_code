import numpy as np

def tau_def(data):
    """
    利用自相关法计算时间延迟 tau。

    参数：
    ----------
    data : 1D array-like
        输入的时间序列数据。

    返回：
    ----------
    tau_value : int
        计算得到的时间延迟 tau，若超过 20，则返回 20。
    """
    # 1. 将输入转换为 numpy 数组，计算序列的均值
    data = np.asarray(data, dtype=float)
    A_ave = np.mean(data)      # 序列平均值
    N = data.size              # 序列长度

    # 2. 自相关阈值：1 - 1/e ≈ 0.63212
    threshold = 1 - 1/np.e

    # 3. 初始化 tau_F（若未找到合适的 t，就保留最大值 20）
    tau_F = 20

    # 4. 从 t=1 到 t=1000 依次计算自相关系数 C(t)
    for t in range(1, 1001):
        # 4.1 计算自相关分子 D 和分母 E
        #    D = sum_{i=0 to N-t-1} (data[i] - A_ave) * (data[i+t] - A_ave)
        #    E = sum_{i=0 to N-t-1} (data[i] - A_ave)^2
        D = np.sum((data[:N-t] - A_ave) * (data[t:] - A_ave))
        E = np.sum((data[:N-t] - A_ave)**2)

        # 4.2 避免分母为零
        if E == 0:
            continue

        # 4.3 计算自相关系数 C(t)
        C_t = D / E

        # 4.4 当 C(t) ≤ 阈值时，记录当前 t 并退出循环
        if C_t <= threshold:
            tau_F = t
            break

    # 5. 若计算得到的 tau_F 大于 20，则返回 20，否则返回 tau_F
    tau_value = tau_F if tau_F <= 20 else 20
    return tau_value
