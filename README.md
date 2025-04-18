### 时频域特征（Time-Frequency Domain Features）

#### 1. 峰峰值均值（Peak-Peak Mean）
**描述**：时间序列中峰（最大值）与谷（最小值）之间垂直长度的算术平均值。  
**计算公式**：  
$$
\text{Peak-Peak Mean} = \frac{1}{N} \sum_{i=1}^{N} (x_{\text{max},i} - x_{\text{min},i})
$$  
其中，$x_{\text{max},i}$ 和 $x_{\text{min},i}$ 为第 $i$ 个时间窗口内的最大和最小值，$N$ 为窗口数。

---

#### 2. 均方值（Mean Square Value）
**描述**：时间序列平方值的算术平均。  
**计算公式**：  
$$
\text{Mean Square Value} = \frac{1}{T} \sum_{t=1}^{T} x(t)^2
$$  
其中，$T$ 为时间序列长度，$x(t)$ 为信号在时间点 $t$ 的值。

---

#### 3. 方差（Variance）
**描述**：反映时间序列的离散程度。  
**计算公式**：  
$$
\text{Variance} = \frac{1}{T} \sum_{t=1}^{T} \left( x(t) - \mu \right)^2
$$  
其中，$\mu$ 为时间序列的均值。

---

#### 4. Hjorth参数：活动性（Activity）
**描述**：反映信号功率，等同于方差。  
**计算公式**：  
$$
\text{Activity} = \text{Var}(x)
$$

---

#### 5. Hjorth参数：移动性（Mobility）
**描述**：估计信号的平均频率，定义为方差的一阶导数与原始方差的平方根之比。  
**计算公式**：  
$$
\text{Mobility} = \sqrt{\frac{\text{Var}(x')}{\text{Var}(x)}}
$$  
其中，$x'$ 为信号的一阶差分。

---

#### 6. Hjorth参数：复杂性（Complexity）
**描述**：反映信号频率变化的复杂度，定义为移动性参数的导数与移动性的比值。  
**计算公式**：  
$$
\text{Complexity} = \frac{\text{Mobility}(x')}{\text{Mobility}(x)}
$$

---

#### 7. 最大功率谱频率（Maximum Power Spectral Frequency）
**描述**：傅里叶变换后功率谱密度最大的频率值。  
**计算步骤**：  
1. 对信号进行傅里叶变换得到功率谱 $P(f)$。  
2. 找到 $P(f)$ 最大值对应的频率 $f_{\text{max}}$。  

---

#### 8. 最大功率谱密度（Maximum Power Spectral Density）
**描述**：功率谱中的最大值，反映信号在频域的能量峰值。  
**计算公式**：  
$$
\text{Max PSD} = \max(P(f))
$$

---

#### 9. 功率总和（Power Sum）
**描述**：所有频率分量的功率之和。  
**计算公式**：  
$$
\text{Power Sum} = \sum_{f} P(f)
$$

---

### 非线性动态系统特征（Non-linear Dynamical System Features）

#### 10. 近似熵（Approximate Entropy, ApEn）
**描述**：衡量时间序列的规律性，值越小表示信号越规则。  
**计算公式**（简化为两参数 $m$ 和 $r$ 的版本）：  
$$
\text{ApEn}(m, r) = \phi^m(r) - \phi^{m+1}(r)
$$  
其中，$\phi^m(r)$ 表示长度为 $m$ 的向量在容限 $r$ 下的相似性度量。

---

#### 11. C0复杂性（C0 Complexity）
**描述**：衡量信号中随机成分的比例。  
**计算公式**：  
1. 将信号分解为规则部分 $x_{\text{reg}}$ 和随机部分 $x_{\text{sto}}$。  
2. 计算随机部分能量占比：  
$$
\text{C0} = \frac{\|x_{\text{sto}}\|^2}{\|x\|^2}
$$

---

#### 12. 相关维（Correlation Dimension）
**描述**：量化相空间中系统状态的分布复杂度，使用关联积分计算。  
**计算公式**：  
$$
D_{\text{corr}} = \lim_{r \to 0} \frac{\log C(r)}{\log r}
$$  
其中，$C(r)$ 为关联积分，表示距离小于 $r$ 的向量对比例。

---

#### 13. Lyapunov指数（Lyapunov Exponent）
**描述**：衡量混沌系统中邻近轨道的指数发散率。  
**计算公式**（最大Lyapunov指数）：  
$$
\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\|\delta x(t)\|}{\|\delta x(0)\|}
$$  
其中，$\delta x(t)$ 为初始扰动的演化。

---

#### 14. Kolmogorov熵（Kolmogorov Entropy）
**描述**：量化系统信息产生或丢失的速率，与混沌程度正相关。  
**计算方式**：通过相空间重构估计熵率。

---

#### 15. 排列熵（Permutation Entropy, PE）
**描述**：基于符号动力学，将时间序列映射为符号序列后计算熵。  
**计算步骤**：  
1. 将时间序列划分为长度为 $m$ 的窗口，生成符号序列。  
2. 计算符号排列的概率分布 $P(\pi)$。  
3. 计算熵：  
$$
\text{PE} = -\sum_{\pi} P(\pi) \log P(\pi)
$$

---

#### 16. 奇异谱熵（Singular Spectrum Entropy）
**描述**：通过SVD分解轨迹矩阵，计算奇异值的熵。  
**计算步骤**：  
1. 重构轨迹矩阵 $X$。  
2. 对 $X$ 进行SVD分解，得到奇异值 $\sigma_i$。  
3. 计算归一化奇异值 $p_i = \sigma_i / \sum \sigma_i$。  
4. 熵计算：  
$$
\text{SSE} = -\sum p_i \log p_i
$$

---

#### 17. Shannon熵（Shannon Entropy）
**描述**：经典信息熵，衡量时间序列的不确定性。  
**计算公式**：  
$$
H = -\sum_{i} p(x_i) \log p(x_i)
$$  
其中，$p(x_i)$ 为信号值 $x_i$ 的概率分布。

---

#### 18. 功率谱熵（Power Spectral Entropy）
**描述**：基于功率谱的Shannon熵，反映频谱复杂性。  
**计算步骤**：  
1. 计算功率谱 $P(f)$。  
2. 归一化功率谱：$\tilde{P}(f) = P(f) / \sum P(f)$。  
3. 熵计算：  
$$
\text{Power Spectral Entropy} = -\sum \tilde{P}(f) \log \tilde{P}(f)
$$

