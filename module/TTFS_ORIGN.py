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
    def __init__(self, units: int, name: str, X_n: float = 1, outputLayer: bool = False, 
                 robustness_params: Dict = {}, input_dim: Optional[int] = None,
                 kernel_regularizer=None, kernel_initializer=None):
        super(SpikingDense, self).__init__()
        self.units = units
        self.B_n = (1 + 0.5) * X_n
        self.outputLayer = outputLayer
        self.t_min_prev = self.t_min = 0
        self.t_max = 1
        self.robustness_params = robustness_params
        self.alpha = torch.ones(units, dtype=torch.float64)
        self.input_dim = input_dim

        # Initialize weights and bias
        if input_dim is not None:
            self.kernel = nn.Parameter(torch.empty(input_dim, units, dtype=torch.float64))
        self.D_i = nn.Parameter(torch.zeros(units, dtype=torch.float64))
        
        # Initialize weights if initializer provided
        if kernel_initializer:
            if isinstance(kernel_initializer, str):
                if kernel_initializer == 'glorot_uniform':
                    nn.init.xavier_uniform_(self.kernel)
                # Add other initializers as needed
            else:
                kernel_initializer(self.kernel)

    def set_params(self, t_min_prev: float, t_min: float):
        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64)
        self.t_min = torch.tensor(t_min, dtype=torch.float64)
        self.t_max = torch.tensor(t_min + self.B_n, dtype=torch.float64)
        return t_min, t_min + self.B_n

    def forward(self, tj):
        if not self.outputLayer:
            output = call_spiking(tj, self.kernel, self.D_i, self.t_min_prev, 
                                self.t_min, self.t_max, self.robustness_params)
        else:
            # Output layer processing
            W_mult_x = torch.matmul(self.t_min - tj, self.kernel)
            self.alpha = self.D_i / (self.t_min - self.t_min_prev)
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
        min_ti = []
        
        for layer in self.layers:
            if isinstance(layer, (SpikingDense, SpikingConv2D)):
                t_max = t_min + max(layer.t_max - layer.t_min, 
                                  10.0 * (layer.t_max - torch.min(x)))
                layer.t_min_prev = t_min_prev
                layer.t_min = t_min
                layer.t_max = t_max
                t_min_prev, t_min = t_min, t_max
            x = layer(x)
            if isinstance(layer, (SpikingDense, SpikingConv2D)):
                min_ti.append(torch.min(x))
                
        return x, min_ti
    
    
    
class SpikingConv2D(nn.Module):
    def __init__(self, filters: int, name: str, X_n: float = 1, padding: str = 'same',
                 kernel_size: tuple = (3, 3), robustness_params: Dict = {},
                 kernel_regularizer=None, kernel_initializer=None):
        super(SpikingConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding.lower()
        self.B_n = (1 + 0.5) * X_n
        self.t_min_prev = self.t_min = 0
        self.t_max = 1
        self.robustness_params = robustness_params.get('time_bits', {})
        self.alpha = torch.ones(filters, dtype=torch.float64)
        
        self.padding_size = kernel_size[0] // 2 if self.padding == 'same' else 0
        
        # Flags for batch normalization
        self.BN = torch.tensor([0])
        self.BN_before_ReLU = torch.tensor([0])
        
        # Initialize D_i with 9 different threshold values
        self.D_i = nn.Parameter(torch.zeros(9, filters, dtype=torch.float64))

    def build(self, input_channels):
        self.kernel = nn.Parameter(
            torch.empty(self.filters, input_channels, *self.kernel_size, dtype=torch.float64))
        if isinstance(self.kernel_initializer, str):
            if self.kernel_initializer == 'glorot_uniform':
                nn.init.xavier_uniform_(self.kernel)

    def set_params(self, t_min_prev: float, t_min: float):
        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64)
        self.t_min = torch.tensor(t_min, dtype=torch.float64)
        self.t_max = torch.tensor(t_min + self.B_n, dtype=torch.float64)
        return t_min, t_min + self.B_n

    def forward(self, tj):
        batch_size = tj.shape[0]
        if self.padding == 'same':
            tj = F.pad(tj, (self.padding_size,) * 4, value=self.t_min)

        # Implementation of conv2d equivalent of extract_patches
        # This part would need careful implementation to match TF exactly
        unfold = nn.Unfold(kernel_size=self.kernel_size, stride=1, padding=0)
        tj_unfolded = unfold(tj)
        
        if self.padding == 'valid' or self.BN != 1 or self.BN_before_ReLU == 1:
            ti = call_spiking(tj_unfolded, self.kernel.view(-1, self.filters), 
                            self.D_i[0], self.t_min_prev, self.t_min, self.t_max, 
                            self.robustness_params)
            # Reshape output
            output_size = tj.shape[2] - self.kernel_size[0] + 1 if self.padding == 'valid' else tj.shape[2]
            ti = ti.view(batch_size, output_size, output_size, self.filters)
        else:
            # Implementation for 9 different thresholds
            # This would need careful implementation to match TF exactly
            pass
            
        return ti