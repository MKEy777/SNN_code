import torch
import torch.nn as nn

# 工具函数

def quantize_tensor(x, min_val, max_val, num_bits):
    """对张量进行量化：把[min_val, max_val]区间映射到[0, 2^num_bits - 1]"""
    scale = (2 ** num_bits - 1) / (max_val - min_val)
    x_scaled = ((x - min_val) * scale).round() / scale + min_val
    return x_scaled

def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    """计算 SpikingDense 层的输出脉冲
    tj:上一层神经元的脉冲时间(形如 [batch, in_features])
    W:当前层的连接权重矩阵
    D_i:当前层的神经元固有延迟(intrinsic delay)
    t_min_prev(是上一层的最小脉冲发放时间) 
    """
    if robustness_params.get('time_bits', 0) != 0:
        tj = quantize_tensor(tj - t_min_prev, t_min_prev, t_min,
                             robustness_params['time_bits']) + t_min_prev
    
    if robustness_params.get('weight_bits', 0) != 0:
        W = quantize_tensor(W, robustness_params['w_min'],
                            robustness_params['w_max'],
                            robustness_params['weight_bits'])

    threshold = t_max - t_min - D_i
    ti = torch.matmul(tj - t_min, W) + threshold + t_min
    ti = torch.where(ti < t_max, ti, t_max)
    
    if robustness_params.get('noise', 0) != 0:
        ti = ti + torch.randn_like(ti) * robustness_params['noise']
    
    return ti

# SpikingDense 层

class SpikingDense(nn.Module):
    def __init__(self, units: int, input_dim: int, outputLayer: bool = False,
                 robustness_params: dict = {}, kernel_initializer='glorot_uniform'):
        super(SpikingDense, self).__init__()
        self.units = units
        self.B_n = 1.5  # 表示“有效时间窗口长度”，用于计算 t_max
        self.outputLayer = outputLayer
        self.robustness_params = robustness_params

        # 初始化权重和参数
        self.kernel = nn.Parameter(torch.empty(input_dim, units, dtype=torch.float32))
        self.D_i = nn.Parameter(torch.zeros(units, dtype=torch.float32))
        
        # 权重初始化
        if kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.kernel)
        else:
            nn.init.normal_(self.kernel, mean=0.0, std=1.0)

    def set_params(self, t_min_prev: float, t_min: float, device):
        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float32, device=device)
        self.t_min = torch.tensor(t_min, dtype=torch.float32, device=device)
        self.t_max = self.t_min + self.B_n  
        self.alpha = torch.ones(self.units, dtype=torch.float32, device=device)
        return t_min, self.t_max.item()


    def forward(self, tj):
        if not self.outputLayer:
            output = call_spiking(tj, self.kernel, self.D_i, self.t_min_prev,
                                  self.t_min, self.t_max, self.robustness_params)
        else:
            W_mult_x = torch.matmul(self.t_min - tj, self.kernel)
            self.alpha = self.D_i / (self.t_min - self.t_min_prev + 1e-10)  
            output = self.alpha * (self.t_min - self.t_min_prev) + W_mult_x
        return output

#  SNN 模型

class SNNModel(nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        t_min_prev, t_min = 0.0, 1.0
        for layer in self.layers:
            if isinstance(layer, SpikingDense):
                device = x.device
                t_min_prev, t_min = layer.set_params(t_min_prev, t_min, device)
            x = layer(x)
        return x