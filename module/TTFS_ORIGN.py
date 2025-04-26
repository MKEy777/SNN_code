# snn_model_definitions.py
import torch
import torch.nn as nn
from typing import Dict, Union, List, Optional

class SpikingDense(nn.Module):
    """
    脉冲神经网络 (SNN) 的脉冲全连接层 (使用 float32)。
    """
    def __init__(self, units: int, name: str, X_n: float = 1, outputLayer: bool = False,
                 robustness_params: Dict = {}, input_dim: Optional[int] = None,
                 kernel_regularizer=None, kernel_initializer=None):
        super(SpikingDense, self).__init__()
        self.units = units
        self.B_n = (1 + 0.5) * X_n
        self.outputLayer = outputLayer
        # 初始化时间参数为 float32
        self.t_min_prev = torch.tensor(0.0, dtype=torch.float32)
        self.t_min = torch.tensor(0.0, dtype=torch.float32)
        self.t_max = torch.tensor(1.0, dtype=torch.float32)
        self.robustness_params = robustness_params
        # Alpha 参数为 float32
        self.alpha = nn.Parameter(torch.ones(units, dtype=torch.float32), requires_grad=False)
        self.input_dim = input_dim
        self.name = name
        self.kernel_initializer_config = kernel_initializer

        # 初始化权重 (kernel) 和偏置 (D_i) 为 float32
        if input_dim is not None:
            # 使用 float32
            self.kernel = nn.Parameter(torch.empty(input_dim, units, dtype=torch.float32))
        # 使用 float32
        self.D_i = nn.Parameter(torch.zeros(units, dtype=torch.float32), requires_grad=True)

        self._initialize_weights()

    def _initialize_weights(self):
        """ Helper function to initialize weights and bias """
        if hasattr(self, 'kernel'):
            kernel_initializer = self.kernel_initializer_config
            if kernel_initializer:
                if isinstance(kernel_initializer, str):
                    if kernel_initializer == 'glorot_uniform':
                        nn.init.xavier_uniform_(self.kernel)
                    else:
                        print(f"Warning: Unknown kernel_initializer string '{kernel_initializer}'. Using xavier_uniform_.")
                        nn.init.xavier_uniform_(self.kernel)
                else:
                    try:
                        kernel_initializer(self.kernel)
                    except Exception as e:
                         print(f"Warning: Failed to apply custom kernel_initializer {e}. Using xavier_uniform_.")
                         nn.init.xavier_uniform_(self.kernel)
            else:
                 nn.init.xavier_uniform_(self.kernel)
        # Bias initialization (already float32)
        # nn.init.zeros_(self.D_i)

    def build(self, input_shape):
        """ Dynamically build the layer (create kernel) if input_dim wasn't known initially. """
        if not hasattr(self, 'kernel'):
            self.input_dim = input_shape[-1]
             # 使用 float32
            self.kernel = nn.Parameter(torch.empty(self.input_dim, self.units, dtype=torch.float32))
            self._initialize_weights()

    def set_params(self, t_min_prev: float, t_min: float):
        """ 设置层的最小和最大脉冲时间边界 (使用 float32)。 """
        # 使用 float32
        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float32)
        self.t_min = torch.tensor(t_min, dtype=torch.float32)
        self.t_max = torch.tensor(t_min + self.B_n, dtype=torch.float32)
        return self.t_min, self.t_max

    def forward(self, tj):
        """ 执行 SpikingDense 层的前向传播 (使用 float32)。 """
        # Ensure input is float32
        tj = tj.to(torch.float32)

        if not hasattr(self, 'kernel'):
             self.build(tj.shape)
             self.kernel = self.kernel.to(tj.device)

        # Move parameters to the correct device and ensure float32
        self.alpha = self.alpha.to(tj.device).to(torch.float32)
        self.t_min_prev = self.t_min_prev.to(tj.device).to(torch.float32)
        self.t_min = self.t_min.to(tj.device).to(torch.float32)
        self.t_max = self.t_max.to(tj.device).to(torch.float32)
        # Ensure kernel and bias are on the correct device (handled by model.to(device))
        # self.kernel = self.kernel.to(tj.device) # Parameter, moved by model.to(device)
        # self.D_i = self.D_i.to(tj.device) # Parameter, moved by model.to(device)

        if self.outputLayer:
            # Ensure calculation uses float32
            W_mult_x = torch.matmul(self.t_min - tj, self.kernel)
            output = W_mult_x + self.D_i
        else:
            output = call_spiking(tj, self.kernel, self.D_i, self.t_min_prev,
                                  self.t_min, self.t_max, self.robustness_params)
        return output

class SNNModel(nn.Module):
    """ 简单的顺序脉冲神经网络模型容器 (使用 float32)。 """
    def __init__(self):
        super(SNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.loss_fn = nn.CrossEntropyLoss()

    def add(self, layer):
        """向模型添加一个层。"""
        # Ensure added layer parameters are float32 (should be by default now)
        self.layers.append(layer.to(torch.float32))

    def forward(self, x):
        """ 执行通过 SNN 所有层的前向传播 (使用 float32)。 """
        # Ensure input is float32
        x = x.to(torch.float32)
        # 使用 float32 初始化时间边界
        t_min_prev = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        t_min = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        min_ti_per_layer = []

        for i, layer in enumerate(self.layers):
            # Ensure layer is float32 (important if layers weren't added via self.add)
            layer = layer.to(torch.float32)
            if isinstance(layer, SpikingDense):
                 # Ensure time parameters passed are standard floats
                 current_t_min, current_t_max = layer.set_params(t_min_prev.item(), t_min.item())
                 x = layer(x) # Pass float32 tensor
                 if torch.is_tensor(x) and x.numel() > 0 and not layer.outputLayer:
                     try:
                         # Convert potential float32 result to standard float for list
                         if torch.all(torch.isfinite(x)):
                              min_val = torch.min(x).item()
                              min_ti_per_layer.append(min_val)
                         else: min_ti_per_layer.append(float('nan'))
                     except RuntimeError: min_ti_per_layer.append(float('inf'))
                 t_min_prev = current_t_min
                 t_min = current_t_max
                 if t_min <= t_min_prev:
                     t_min = t_min_prev + 1e-6 # Use float32 epsilon?
            else:
                 x = layer(x.to(torch.float32)) # Ensure input to non-SNN layers is float32

        return x, min_ti_per_layer

def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    """ 确定标准脉冲层中输出脉冲时间的核心计算 (使用 float32)。 """
    # Ensure inputs are float32
    tj = tj.to(torch.float32)
    W = W.to(torch.float32)
    D_i = D_i.to(torch.float32)
    t_min_prev = t_min_prev.to(torch.float32)
    t_min = t_min.to(torch.float32)
    t_max = t_max.to(torch.float32)

    # --- 量化 (如果使用，确保在 float32 上操作) ---
    if robustness_params.get('time_bits', 0) > 0:
        relative_tj = tj - t_min_prev
        quant_max_val = t_min - t_min_prev
        epsilon = torch.tensor(1e-9, dtype=torch.float32, device=tj.device)
        safe_quant_max_val = torch.where(quant_max_val <= epsilon, epsilon, quant_max_val)
        quantized_relative_tj = quantize_tensor(relative_tj,
                                                torch.tensor(0.0, dtype=torch.float32, device=tj.device),
                                                safe_quant_max_val,
                                                robustness_params['time_bits'])
        tj = quantized_relative_tj + t_min_prev

    if robustness_params.get('weight_bits', 0) > 0:
        w_min = robustness_params.get('w_min', torch.min(W).item())
        w_max = robustness_params.get('w_max', torch.max(W).item())
        W = quantize_tensor(W, torch.tensor(w_min, dtype=torch.float32, device=W.device),
                            torch.tensor(w_max, dtype=torch.float32, device=W.device),
                            robustness_params['weight_bits'])

    # --- 核心脉冲时间计算 (使用 float32) ---
    threshold = t_max - t_min - D_i
    relative_tj = tj - t_min
    W = W.to(relative_tj.device)
    threshold = threshold.to(relative_tj.device)
    t_min = t_min.to(relative_tj.device)
    t_max = t_max.to(relative_tj.device)
    ti_relative = torch.matmul(relative_tj, W) + threshold
    ti = ti_relative + t_min

    # --- 裁剪 (使用 float32) ---
    ti = torch.clamp(ti, min=t_min, max=t_max)

    # --- 噪声注入 (可选, 使用 float32) ---
    if robustness_params.get('noise', 0) != 0:
        noise_std = torch.tensor(robustness_params['noise'], dtype=torch.float32, device=ti.device)
        noise = torch.randn_like(ti) * noise_std # randn_like creates float32 if ti is float32
        ti = ti + noise
        ti = torch.clamp(ti, min=t_min, max=t_max)

    return ti

def quantize_tensor(x, min_val, max_val, num_bits):
    """ 对张量执行简单的线性量化 (输入/输出 dtype 保持不变)。 """
    # 量化函数本身不强制类型，它使用输入 x 的类型
    if num_bits <= 0: return x
    # Ensure min/max vals match tensor type for calculations
    min_val = torch.as_tensor(min_val, dtype=x.dtype, device=x.device)
    max_val = torch.as_tensor(max_val, dtype=x.dtype, device=x.device)
    if torch.any(max_val <= min_val): return x
    q_levels = 2**num_bits
    # Ensure scale calculation maintains precision / correct type
    scale = (q_levels - 1) / (max_val - min_val)
    x_shifted = x - min_val
    x_scaled = x_shifted * scale
    x_rounded = torch.round(x_scaled)
    x_rescaled = x_rounded / scale
    x_quantized = x_rescaled + min_val
    x_clamped = torch.clamp(x_quantized, min=min_val, max=max_val)
    return x_clamped