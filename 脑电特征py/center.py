def center(data, f):
    """
    计算data数组累积到总和一半时对应的f值

    参数:
        data: 输入数据数组
        f: 与data对应的一维数组

    返回:
        out: 当data累积和达到或超过总和一半时对应的f值
    """
    # 检查输入是否为空
    if len(data) == 0:
        raise ValueError("data数组不能为空")

    # 计算data数组总和的一半
    power = sum(data) * 0.5

    # 初始化累加器
    add_power = 0

    # 遍历data数组
    for i in range(1, len(data) + 1):
        # 累加当前元素
        add_power = data[i - 1] + add_power
        # 如果累加和大于或等于power，跳出循环
        if add_power > power or add_power == power:
            break

    # 返回对应的f值
    out = f[i - 1]
    return out