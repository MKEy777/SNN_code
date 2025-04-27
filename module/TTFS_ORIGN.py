import torch
import torch.nn as nn
from typing import Dict, Union, List, Optional
import warnings

# 设置默认浮点类型为 float32
torch.set_default_dtype(torch.float32)

def quantize_tensor(x, min_val, max_val, num_bits):
    """
    将输入 x 在[min_val, max_val]范围内量化为 num_bits 位
    """
    if num_bits <= 0 or num_bits >= 16:
        # 如果不需要或位数过高，则直接返回原始张量
        return x

    qmin = 0.
    qmax = float(2**num_bits - 1)

    # 将 min_val, max_val 转为张量并放在相同设备
    min_val_t = torch.as_tensor(min_val, dtype=x.dtype, device=x.device)
    max_val_t = torch.as_tensor(max_val, dtype=x.dtype, device=x.device)

    if torch.equal(min_val_t, max_val_t):
        # 如果区间长度为零则直接裁剪
        return torch.clamp(x, min=min_val_t, max=max_val_t)

    # 计算量化尺度
    scale = (max_val_t - min_val_t) / (qmax - qmin)

    if torch.abs(scale) < 1e-9:
       warnings.warn(f"Quantization scale is close to zero ({scale.item()}). Clamping input between {min_val_t.item()} and {max_val_t.item()}.")
       return torch.clamp(x, min=min_val_t, max=max_val_t)

    # 计算零点
    zero_point_float = qmin - min_val_t / scale
    zero_point = torch.round(zero_point_float)
    zero_point_clamped = torch.clamp(zero_point, qmin, qmax).to(torch.int64)

    try:
        # 使用 PyTorch 假量化函数
        scale_32 = scale.to(torch.float32)
        x_quantized = torch.fake_quantize_per_tensor_affine(
            x, scale_32, zero_point_clamped, int(qmin), int(qmax)
        )
    except RuntimeError as e:
         warnings.warn(f"torch.fake_quantize_per_tensor_affine encountered an error: {e}. Falling back.")
         x_quantized = torch.clamp(x, min=min_val_t, max=max_val_t)

    return x_quantized


class SpikingDense(nn.Module):
    """
    带脉冲时序编码的全连接层（Dense）
    - 可选择作为输出层(outputLayer)
    - 支持权重和时间的量化与噪声注入
    """
    def __init__(
        self, units: int, name: str, X_n: float = 1,
        outputLayer: bool = False, robustness_params: Dict = {},
        input_dim: Optional[int] = None, kernel_regularizer=None,
        kernel_initializer=None
    ):
        super(SpikingDense, self).__init__()
        self.units = units
        self.name = name  # 层的名称
        self.outputLayer = outputLayer  # 是否为输出层
        # B_n 用于定义时间窗口宽度: (1 + 0.5) * X_n
        self.register_buffer('B_n', torch.tensor((1 + 0.5) * X_n))
        # 初始化时间界限，用于脉冲编码时的前后时间管理
        self.t_min_prev_cpu = torch.tensor(0.0)
        self.t_min_cpu = torch.tensor(1.0)
        self.t_max_cpu = self.t_min_cpu + self.B_n

        self.robustness_params = robustness_params  # 鲁棒性参数: time_bits, weight_bits, noise
        # alpha 通常用于学习率调节等，但这里不参与梯度更新
        self.alpha = nn.Parameter(torch.ones(units), requires_grad=False)
        self.input_dim = input_dim
        self.kernel_initializer_config = kernel_initializer

        # D_i: 偏置项，支持时序偏移
        self.D_i = nn.Parameter(torch.zeros(units), requires_grad=True)

        # 如果已知输入维度，则立即初始化权重矩阵
        if input_dim is not None:
            self.kernel = nn.Parameter(torch.empty(input_dim, units))
            self._initialize_weights()

    def _initialize_weights(self):
        """
        根据指定的初始化器(kernel_initializer)进行权重初始化，
        默认使用 Xavier 均匀分布
        """
        if hasattr(self, 'kernel'):
            init_cfg = self.kernel_initializer_config
            if init_cfg:
                if isinstance(init_cfg, str) and init_cfg == 'glorot_uniform':
                    nn.init.xavier_uniform_(self.kernel)
                else:
                    try:
                        init_cfg(self.kernel)
                    except Exception as e:
                        warnings.warn(f"Layer '{self.name}': Failed to apply custom initializer {e}. Using xavier_uniform_.")
                        nn.init.xavier_uniform_(self.kernel)
            else:
                nn.init.xavier_uniform_(self.kernel)

    def build(self, input_shape):
        """
        动态在第一次前向传递时根据输入形状构建权重矩阵
        """
        if not hasattr(self, 'kernel'):
            self.input_dim = input_shape[-1]
            device = self.D_i.device
            self.kernel = nn.Parameter(torch.empty(self.input_dim, self.units, device=device))
            print(f"Layer '{self.name}' dynamically built on device: {device} with dtype: {self.kernel.dtype}")
            self._initialize_weights()

    def update_time_bounds(self, t_min_prev: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor):
        """
        在训练时根据上一层输出脉冲时间更新当前层的时间范围
        """
        self.t_min_prev_cpu = t_min_prev.cpu().detach()
        self.t_min_cpu = t_min.cpu().detach()
        self.t_max_cpu = t_max.cpu().detach()

    def forward(self, tj):
        """
        前向计算：
          - 输出层: 直接线性变换
          - 隐藏层: 调用 call_spiking 实现脉冲时序计算
        返回: (输出张量, 最小脉冲时间估计)
        """
        device = self.D_i.device
        tj = tj.to(device, dtype=torch.float32)

        # 若权重未初始化，则先构建
        if not hasattr(self, 'kernel'):
            self.build(tj.shape)

        # 将 CPU 存储的时间边界拉到当前设备
        t_min_prev_dev = self.t_min_prev_cpu.to(device)
        t_min_dev = self.t_min_cpu.to(device)
        t_max_dev = self.t_max_cpu.to(device)

        min_ti_output = None

        if self.outputLayer:
            # 输出层: 简化的线性时序映射
            W_mult_x = torch.matmul(t_min_dev - tj, self.kernel)
            output = W_mult_x + self.D_i
        else:
            # 隐藏层: 调用脉冲计算核心函数
            output = call_spiking(
                tj, self.kernel, self.D_i,
                t_min_prev_dev, t_min_dev, t_max_dev,
                self.robustness_params
            )
            # 记录最小脉冲时间（用于动态调整时间边界）
            if torch.is_tensor(output) and output.numel() > 0:
                try:
                    min_ti_output = torch.min(output.detach()).clone()
                except RuntimeError:
                    min_ti_output = torch.tensor(float('inf'), device=device)
            else:
                min_ti_output = torch.tensor(float('inf'), device=device)

        return output, min_ti_output


class SNNModel(nn.Module):
    """
    构建脉冲神经网络模型，按顺序管理多个 SpikingDense 层
    """
    def __init__(self):
        super(SNNModel, self).__init__()
        self.layers_list = nn.ModuleList()

    def add(self, layer):
        # 添加层到模型
        self.layers_list.append(layer.to(dtype=torch.float32))

    def forward(self, x):
        # 确定模型当前运行设备
        current_device = x.device if len(list(self.parameters())) == 0 else next(self.parameters()).device
        x = x.to(current_device, dtype=torch.float32)

        min_ti_list = []
        # 依次调用每层的前向输出
        for layer in self.layers_list:
            layer = layer.to(current_device)
            if isinstance(layer, SpikingDense):
                x, min_ti = layer(x)
                if not layer.outputLayer:
                    min_ti_list.append(min_ti if min_ti is not None else torch.tensor(float('inf'), device=current_device))
            else:
                x = layer(x)
        return x, min_ti_list


def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    """
    脉冲计算核心函数：
      1. 时间量化 (time_bits)
      2. 权重量化 (weight_bits)
      3. 线性时序映射 + 偏置
      4. 添加高斯噪声模拟不确定性
    """
    device = tj.device
    # 1) 时间量化
    time_bits = robustness_params.get('time_bits', 0)
    if time_bits > 0:
        rel_tj = tj - t_min_prev
        quant_max = t_min - t_min_prev
        rel_tj_q = quantize_tensor(rel_tj, 0.0, quant_max, time_bits)
        tj = rel_tj_q + t_min_prev

    # 2) 权重量化
    weight_bits = robustness_params.get('weight_bits', 0)
    if weight_bits > 0:
        w_min = float(robustness_params.get('w_min', torch.min(W.detach()).item()))
        w_max = float(robustness_params.get('w_max', torch.max(W.detach()).item()))
        W = quantize_tensor(W, w_min, w_max, weight_bits)

    # 3) 计算脉冲时间 ti
    threshold = t_max - t_min - D_i
    rel_tj2 = tj - t_min
    ti_rel = torch.matmul(rel_tj2, W)
    ti = ti_rel + threshold + t_min
    ti = torch.clamp(ti, min=t_min, max=t_max)

    # 4) 添加噪声
    noise_std = robustness_params.get('noise', 0)
    if noise_std != 0:
        noise = torch.randn_like(ti) * noise_std
        ti = torch.clamp(ti + noise, min=t_min, max=t_max)

    return ti
