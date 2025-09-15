import torch
import torch.nn as nn
from typing import Dict, Union, List, Optional, Tuple
import warnings

torch.set_default_dtype(torch.float32)
EPSILON = 1e-9

def call_spiking_layer(tj: torch.Tensor, W: torch.Tensor, D_i: torch.Tensor,
                       t_min_prev: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor,
                       robustness_params: Dict = {}) -> torch.Tensor:
    """
    参数:
        tj (torch.Tensor): 输入脉冲时间，形状 (batch_size, input_dim)。
        W (torch.Tensor): 权重矩阵，形状 (input_dim, units)。
        D_i (torch.Tensor): 可训练的阈值调整参数，形状 (units,)。
        t_min_prev (torch.Tensor): 上一层时间下界 (标量)。
        t_min (torch.Tensor): 当前层时间下界 (标量)。
        t_max (torch.Tensor): 当前层时间上界 (标量)。
        robustness_params (Dict): 噪声/量化占位 (此处未实现)。

    返回:
        torch.Tensor: 输出脉冲时间 ti，形状 (batch_size, units)。未触发脉冲的值截断为 t_max。
    """
    current_device = tj.device
    # 确保参数在正确设备和数据类型上
    W = W.to(dtype=torch.float32)
    D_i = D_i.to(dtype=torch.float32)
    t_min_prev = t_min_prev.to(dtype=torch.float32, device=current_device)
    t_min = t_min.to(dtype=torch.float32, device=current_device)
    t_max = t_max.to(dtype=torch.float32, device=current_device)
    # 计算阈值：t_max - t_min - D_i
    threshold = t_max - t_min - D_i  # (units,)
    # 计算脉冲时间：W*(tj - t_min) + threshold + t_min
    ti = torch.matmul(tj - t_min, W) + threshold + t_min  # (batch_size, units)
    # 将大于 t_max 的值截断
    ti = torch.where(ti < t_max, ti, t_max)

    return ti

class SpikingDense(nn.Module):
    def __init__(self, units: int, name: str, outputLayer: bool = False,
                 robustness_params: Dict = {}, input_dim: Optional[int] = None,
                 kernel_regularizer=None,
                 kernel_initializer='glorot_uniform'):
        super(SpikingDense, self).__init__()
        self.units = units
        self.name = name
        self.outputLayer = outputLayer
        self.robustness_params = robustness_params
        self.input_dim = input_dim
        self.initializer = kernel_initializer

        # 可训练参数
        self.D_i = nn.Parameter(torch.zeros(units, dtype=torch.float32))  # 阈值调整
        self.kernel = None  # 权重矩阵 W

        # 注册时间边界缓冲区
        self.register_buffer('t_min_prev', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('t_min', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(1.0, dtype=torch.float32))

        self.built = False
        if self.input_dim is not None:
            self.kernel = nn.Parameter(torch.empty(self.input_dim, self.units, dtype=torch.float32))
            self._initialize_weights()
            self.built = True

    def _initialize_weights(self):
        """根据 self.initializer 初始化 kernel 权重。"""
        init_config = self.initializer
        if init_config:
            if isinstance(init_config, str):
                if init_config == 'glorot_uniform':
                    nn.init.xavier_uniform_(self.kernel)
                elif init_config == 'glorot_normal':
                    nn.init.xavier_normal_(self.kernel)
                else:
                    nn.init.xavier_uniform_(self.kernel)
            elif callable(init_config):
                init_config(self.kernel)
            else:
                nn.init.xavier_uniform_(self.kernel)
        else:
            nn.init.xavier_uniform_(self.kernel)

        # 将 D_i 清零
        with torch.no_grad():
            self.D_i.zero_()

    def build(self, input_shape: Union[torch.Size, Tuple, List]):
        """
        动态创建权重（如果尚未创建）。
        """
        if self.built:
            return
        if isinstance(input_shape, (tuple, list)):
            input_shape = torch.Size(input_shape)
        in_dim = input_shape[-1]

        self.input_dim = in_dim

        # 从已有参数或缓冲区获取设备信息
        try:
            current_device = self.D_i.device
        except AttributeError:
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建 kernel 参数
        self.kernel = nn.Parameter(
            torch.empty(self.input_dim, self.units, dtype=torch.float32, device=current_device)
        )
        self._initialize_weights()
        self.built = True

    def set_time_params(self, t_min_prev: Union[float, torch.Tensor], t_min: Union[float, torch.Tensor], t_max: Union[float, torch.Tensor]):
        """
        设置当前层的时间边界。在训练循环中，forward 前调用。
        """
        buffer_device = self.t_min_prev.device  # 缓冲区的设备
        if not isinstance(t_min_prev, torch.Tensor):
            t_min_prev = torch.tensor(t_min_prev, dtype=torch.float32)
        if not isinstance(t_min, torch.Tensor):
            t_min = torch.tensor(t_min, dtype=torch.float32)
        if not isinstance(t_max, torch.Tensor):
            t_max = torch.tensor(t_max, dtype=torch.float32)

        # 原地更新缓冲区值
        self.t_min_prev.copy_(t_min_prev.to(buffer_device))
        self.t_min.copy_(t_min.to(buffer_device))
        self.t_max.copy_(t_max.to(buffer_device))

    def forward(self, tj: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播。

        参数:
            tj (torch.Tensor): 输入脉冲时间 (batch_size, input_dim)，应 <= 上一层 t_max。

        返回:
            Tuple:
                - output (torch.Tensor): 隐藏层输出脉冲时间或输出层的逻辑值。
                - min_ti (Optional[torch.Tensor]): 本层最小脉冲时间（仅隐藏层），用于更新 t_max。
        """
        if not self.built:
            self.build(tj.shape)

        current_device = self.D_i.device
        tj = tj.to(current_device, dtype=torch.float32)

        t_min_prev_dev = self.t_min_prev
        t_min_dev = self.t_min
        t_max_dev = self.t_max

        min_ti_output = None

        if self.outputLayer:
            # --- 输出层逻辑 ---
            W_mult_x = torch.matmul(t_min_dev - tj, self.kernel)

            time_diff = t_min_dev - t_min_prev_dev
            safe_time_diff = torch.where(time_diff == 0, torch.tensor(EPSILON, device=current_device), time_diff)
            # 计算 alpha，基于 D_i
            alpha = self.D_i / safe_time_diff

            # 计算最终输出（逻辑值）
            output = alpha * time_diff + W_mult_x
            min_ti_output = None
        else:
            # --- 隐藏层逻辑 ---
            output = call_spiking_layer(tj, self.kernel, self.D_i, t_min_prev_dev,
                                        t_min_dev, t_max_dev, self.robustness_params)

            # 计算小于 t_max 的最小脉冲时间，用于下层 t_max 更新
            with torch.no_grad():
                mask = torch.isfinite(output) & (output < t_max_dev)
                spikes = output[mask]
                if spikes.numel() > 0:
                    min_ti_output = torch.min(spikes).detach().unsqueeze(0)
                else:
                    min_ti_output = t_max_dev.clone().detach().unsqueeze(0)

        output = output.to(current_device)
        if min_ti_output is not None:
            min_ti_output = min_ti_output.to(current_device)

        return output, min_ti_output

class SNNModel(nn.Module):
    """
    由多个 SpikingDense 层组成的时空神经网络模型。
    """
    def __init__(self):
        super(SNNModel, self).__init__()
        self.layers_list = nn.ModuleList()

    def add(self, layer: nn.Module):
        """向模型中添加层。"""
        self.layers_list.append(layer.to(dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        模型前向传播。在调用前需外部设置各层时间参数。

        参数:
            x (torch.Tensor): 输入数据（如脉冲时间编码）。

        返回:
            Tuple:
                - final_output (torch.Tensor): 最后一层输出（逻辑值）。
                - min_ti_list (List[Optional[torch.Tensor]]): 每个隐藏层的最小脉冲时间列表。
        """
        current_input = x
        min_ti_list: List[Optional[torch.Tensor]] = []

        # 确定目标设备
        target_device = x.device
        if self.layers_list:
            try:
                first_param = next(self.parameters())
                target_device = first_param.device
            except StopIteration:
                try:
                    first_buffer = next(self.buffers())
                    target_device = first_buffer.device
                except StopIteration:
                    pass

        current_input = current_input.to(target_device, dtype=torch.float32)

        for i, layer in enumerate(self.layers_list):
            layer = layer.to(target_device)

            if isinstance(layer, SpikingDense):
                current_input, min_ti = layer(current_input)
                if not layer.outputLayer:
                    min_ti_list.append(min_ti)
            elif isinstance(layer, nn.Flatten):
                current_input = layer(current_input)
            else:
                current_input = layer(current_input)

        final_output = current_input
        return final_output, min_ti_list