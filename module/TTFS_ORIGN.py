import torch
import torch.nn as nn
from typing import Dict, Union, List, Optional, Tuple
import warnings


torch.set_default_dtype(torch.float32)
# 用于安全除法的小 epsilon 值
EPSILON = 1e-9 # Keep EPSILON as it might be used in core logic beyond checks

def quantize_tensor_like_tf(x, min_val, max_val, num_bits):
    """
    Args:
        x (torch.Tensor): 输入张量 (float32)。
        min_val (float): 量化范围最小值。
        max_val (float): 量化范围最大值。
        num_bits (int): 量化位数。

    Returns:
        torch.Tensor: 量化后的张量 (float32)。
    """
    # Removed: if num_bits <= 0 or num_bits >= 16: return x

    qmin = 0.
    qmax = float(2**num_bits - 1)

    min_val_t = torch.tensor(min_val, dtype=torch.float32, device=x.device)
    max_val_t = torch.tensor(max_val, dtype=torch.float32, device=x.device)

    # Removed: if torch.le(max_val_t, min_val_t): warnings.warn(...) return torch.clamp(...)

    scale = (max_val_t - min_val_t) / (qmax - qmin)
    # Removed: if torch.abs(scale) < EPSILON: warnings.warn(...) return torch.clamp(...)

    zero_point_float = qmin - min_val_t / scale
    zero_point_clamped = torch.clamp(torch.round(zero_point_float), qmin, qmax)
    zero_point_int = zero_point_clamped.to(torch.int64)

    # Removed: try-except block, fallback to clamp
    qmin_int = int(qmin)
    qmax_int = int(qmax)
    x_quantized = torch.fake_quantize_per_tensor_affine(x, scale.item(), zero_point_int.item(),
                                                       qmin_int, qmax_int)

    return x_quantized

def call_spiking_pytorch(tj: torch.Tensor, W: torch.Tensor, D_i: torch.Tensor,
                         t_min_prev: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor,
                         robustness_params: Dict) -> torch.Tensor:
    """
    B_i=1, A_i=0, tau_c=1

    Args:
        tj (torch.Tensor): 输入脉冲时间 (batch_size, input_dim)，应 <= 上一层的 t_max。
        W (torch.Tensor): 权重矩阵 (input_dim, units)。
        D_i (torch.Tensor): 可训练的延迟/阈值调整 (units,)。对应论文中定义的阈值vartheta的一部分。
        t_min_prev (torch.Tensor): 上一层的时间下界 (标量)。
        t_min (torch.Tensor): 当前层的时间下界 (标量)。
        t_max (torch.Tensor): 当前层的时间上界 (标量)。
        robustness_params (Dict): 鲁棒性参数 (量化位数、噪声)。

    Returns:
        torch.Tensor: 输出脉冲时间 ti (batch_size, units)。非脉冲被设置为 t_max。
    """
    current_device = tj.device
    W = W.to(dtype=torch.float32)
    D_i = D_i.to(dtype=torch.float32)
    t_min_prev = t_min_prev.to(dtype=torch.float32, device=current_device)
    t_min = t_min.to(dtype=torch.float32, device=current_device)
    t_max = t_max.to(dtype=torch.float32, device=current_device)

    # --- 时间量化 (可选) ---
    time_bits = robustness_params.get('time_bits', 0)
    if time_bits != 0:
        quant_time_max_val = t_min - t_min_prev
        # Removed: Check if quant_time_max_val > 0
        relative_tj = tj - t_min_prev
        quantized_relative_tj = quantize_tensor_like_tf(relative_tj, 0.0, quant_time_max_val.item(), time_bits)
        tj = t_min_prev + quantized_relative_tj

    # --- 权重量化 (可选) ---
    weight_bits = robustness_params.get('weight_bits', 0)
    if weight_bits != 0:
        w_min = float(robustness_params.get('w_min', torch.min(W.detach()).item()))
        w_max = float(robustness_params.get('w_max', torch.max(W.detach()).item()))
        # Removed: Check if w_max > w_min
        W = quantize_tensor_like_tf(W, w_min, w_max, weight_bits)

    # --- 计算脉冲时间 (基于论文 B1-model, A_i=0, tau_c=1) ---
    threshold = t_max - t_min - D_i # (units,)
    ti_theoretical = torch.matmul(tj - t_min, W) + threshold + t_min

    # --- 添加噪声 (可选) ---
    noise_std_val = robustness_params.get('noise', 0)
    if noise_std_val != 0:
        noise_std = torch.tensor(noise_std_val, device=current_device, dtype=torch.float32)
        ti_theoretical = ti_theoretical + torch.randn_like(ti_theoretical) * noise_std

    # --- 钳位到 t_max ---
    ti = torch.where(ti_theoretical < t_max, ti_theoretical, t_max)

    return ti


class SpikingDense(nn.Module):
    """
    SpikingDense 层实现，支持可训练的延迟/阈值参数 D_i 和权重 W。
    """
    def __init__(self, units: int, name: str, outputLayer: bool = False,
                 robustness_params: Dict = {}, input_dim: Optional[int] = None,
                 kernel_regularizer=None, kernel_initializer=None):
        super(SpikingDense, self).__init__()
        self.units = units
        self.name = name
        self.outputLayer = outputLayer
        self.robustness_params = robustness_params
        self.input_dim = input_dim
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer

        self.D_i = nn.Parameter(torch.zeros(units, dtype=torch.float32))
        self.kernel = None

        if self.outputLayer:
            self.A_i = nn.Parameter(torch.zeros(units, dtype=torch.float32))
        else:
            self.A_i = None

        self.register_buffer('t_min_prev', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('t_min', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(1.0, dtype=torch.float32))

        self.built = False
        # Removed: Check on input_dim type/value during init
        if self.input_dim is not None:
             self.kernel = nn.Parameter(torch.empty(self.input_dim, self.units, dtype=torch.float32))
             self._initialize_weights()
             self.built = True

    def _initialize_weights(self):
        """根据 self.initializer 配置初始化 kernel 权重。"""
        # Removed: Check if self.kernel is None

        init_config = self.initializer
        if init_config:
            if isinstance(init_config, str):
                if init_config == 'glorot_uniform':
                    nn.init.xavier_uniform_(self.kernel)
                elif init_config == 'glorot_normal':
                     nn.init.xavier_normal_(self.kernel)
                # Removed: else block with warning for unknown initializer
                else:
                    # Defaulting to xavier_uniform if unknown string
                    nn.init.xavier_uniform_(self.kernel)
            elif callable(init_config):
                # Removed: try-except block for custom initializer
                init_config(self.kernel)
            # Removed: else block with warning for invalid type
            else:
                # Defaulting if invalid type
                nn.init.xavier_uniform_(self.kernel)
        else:
            nn.init.xavier_uniform_(self.kernel)

        with torch.no_grad():
            self.D_i.zero_()
        if self.A_i is not None:
             with torch.no_grad():
                  self.A_i.zero_()

    def build(self, input_shape: Union[torch.Size, Tuple, List]):
        """如果权重尚未创建，则根据第一次输入的形状动态创建权重。"""
        if self.built:
            return
        # Removed: Type/length checks for input_shape
        if isinstance(input_shape, (tuple, list)):
             input_shape = torch.Size(input_shape)
        in_dim = input_shape[-1]
        # Removed: Check comparing in_dim with self.input_dim
        # Removed: Check on in_dim value
        self.input_dim = in_dim

        # Removed: Check if D_i is initialized
        current_device = self.D_i.device
        self.kernel = nn.Parameter(torch.empty(self.input_dim, self.units, dtype=torch.float32, device=current_device))

        self._initialize_weights()
        self.built = True

    def set_params(self, t_min_prev: Union[float, torch.Tensor], t_min: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        更新层的时间下界 (t_min_prev, t_min)。
        t_max 将由训练循环根据 min_ti 动态计算和设置。

        Args:
            t_min_prev: *前*一层的时间下界 (或初始值 0.0)。
            t_min: 为*此*层计算的时间下界。

        Returns:
            torch.Tensor: 设置后的当前层的 t_min 值 (已分离梯度)。
        """
        if not isinstance(t_min_prev, torch.Tensor):
            t_min_prev = torch.tensor(t_min_prev, dtype=torch.float32)
        if not isinstance(t_min, torch.Tensor):
            t_min = torch.tensor(t_min, dtype=torch.float32)
        # Removed: Dimension check warning
        # Removed: Flattening if not scalar
        # t_min_prev = t_min_prev.flatten()[0]
        # t_min = t_min.flatten()[0]

        buffer_device = self.t_min_prev.device
        self.t_min_prev.copy_(t_min_prev.to(buffer_device))
        self.t_min.copy_(t_min.to(buffer_device))

        return self.t_min.clone().detach()

    def forward(self, tj: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        层的前向传播逻辑。

        Args:
            tj (torch.Tensor): 输入脉冲时间 (batch_size, input_dim)。应 <= 上一层的 t_max。

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - output (torch.Tensor): 输出脉冲时间 (隐藏层) 或 logits (输出层)。
                - min_ti (Optional[torch.Tensor]): 隐藏层计算出的最小有限脉冲时间，否则为 None。
        """
        # Removed: Check if not built (moved build call earlier)
        if not self.built:
            self.build(tj.shape)
        # Removed: Check if kernel is None

        current_device = self.D_i.device
        tj = tj.to(current_device, dtype=torch.float32)

        t_min_prev_dev = self.t_min_prev
        t_min_dev = self.t_min
        t_max_dev = self.t_max

        min_ti_output = None

        if self.outputLayer:
            # Removed: Check if self.A_i is None
            weighted_inputs = torch.matmul(t_min_dev - tj, self.kernel)
            time_diff = t_min_dev - t_min_prev_dev
            bias_term = self.A_i * time_diff
            output = weighted_inputs + bias_term
            min_ti_output = None
        else:
            output = call_spiking_pytorch(tj, self.kernel, self.D_i, t_min_prev_dev,
                                          t_min_dev, t_max_dev, self.robustness_params)

            finite_spikes_mask = torch.isfinite(output) & (output < t_max_dev)
            finite_spikes = output[finite_spikes_mask]
            if finite_spikes.numel() > 0:
                min_ti_output = torch.min(finite_spikes).detach()
            else:
                 min_ti_output = t_max_dev.clone().detach()

        output = output.to(current_device)
        if min_ti_output is not None:
             min_ti_output = min_ti_output.to(current_device)

        return output, min_ti_output


class SNNModel(nn.Module):
    """
    顺序 SNN 模型，由层组成 (例如 SpikingDense)。
    """
    def __init__(self):
        super(SNNModel, self).__init__()
        self.layers_list = nn.ModuleList()

    def add(self, layer: nn.Module):
        """向模型添加层。"""
        # Removed: Type check for layer
        self.layers_list.append(layer.to(dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        模型的前向传播。

        Args:
            x (torch.Tensor): 输入数据 (例如，编码后的脉冲时间)。

        Returns:
            Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
                - final_output (torch.Tensor): 最后一层的输出 (logits)。
                - min_ti_list (List[Optional[torch.Tensor]]): 包含每个*隐藏* SpikingDense 层
                  计算出的最小有限脉冲时间的列表 (可能是 t_max)。
        """
        current_input = x
        min_ti_list = []

        target_device = x.device
        if len(self.layers_list) > 0:
             try:
                  first_param = next(self.parameters())
                  target_device = first_param.device
             except StopIteration:
                  try:
                      first_buffer = next(self.buffers())
                      target_device = first_buffer.device
                  except StopIteration:
                      pass # Keep device as x.device if no params/buffers

        current_input = current_input.to(target_device, dtype=torch.float32)

        for i, layer in enumerate(self.layers_list):
            layer = layer.to(target_device)

            if isinstance(layer, SpikingDense):
                current_input, min_ti = layer(current_input)
                if not layer.outputLayer:
                    min_ti_list.append(min_ti)
            else:
                current_input = layer(current_input)

        final_output = current_input
        return final_output, min_ti_list