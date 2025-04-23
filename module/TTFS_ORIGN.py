import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List, Optional

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


class SpikingConv2D(nn.Module):
    def __init__(self, filters: int, name: str, X_n: float = 1, padding: str = 'same',
                 kernel_size: tuple = (3, 3), robustness_params: Dict = {},
                 kernel_regularizer=None, kernel_initializer=None):
        super(SpikingConv2D, self).__init__()
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding.lower()
        self.B_n = (1 + 0.5) * X_n
        self.t_min_prev = self.t_min = 0
        self.t_max = 1
        self.robustness_params = robustness_params
        self.alpha = torch.ones(filters, dtype=torch.float64)
        
        self.padding_size = kernel_size[0] // 2 if self.padding == 'same' else 0
        
        self.BN = torch.tensor([0])
        self.BN_before_ReLU = torch.tensor([0])
        self.D_i = nn.Parameter(torch.zeros(9, filters, dtype=torch.float64))

    def forward(self, tj):
        batch_size, channels, height, width = tj.shape
        logging.info(f"Input tensor shape: {tj.shape}")
        
        if self.padding == 'same':
            tj = F.pad(tj, (self.padding_size,) * 4, value=self.t_min)
            logging.info(f"After padding shape: {tj.shape}")

        # 计算输出尺寸
        if self.padding == 'same':
            out_height = height
            out_width = width
        else:
            out_height = height - self.kernel_size[0] + 1
            out_width = width - self.kernel_size[1] + 1
        
        logging.info(f"Expected output dimensions: h={out_height}, w={out_width}")

        # 使用 unfold 提取补丁
        unfold = nn.Unfold(kernel_size=self.kernel_size, 
                        stride=1, 
                        padding=0)  # 修改为始终使用 padding=0
        patches = unfold(tj)
        logging.info(f"Unfolded patches shape: {patches.shape}")
        
        # 重塑补丁
        patches = patches.transpose(1, 2).contiguous()
        logging.info(f"Transposed patches shape: {patches.shape}")
        
        # 展平卷积核
        kernel_flat = self.kernel.view(self.filters, -1)
        logging.info(f"Flattened kernel shape: {kernel_flat.shape}")

        if self.padding == 'valid' or self.BN != 1 or self.BN_before_ReLU == 1:
            logging.info(f"Input to call_spiking - patches: {patches.shape}, kernel: {kernel_flat.t().shape}")
            
            ti = call_spiking(patches, kernel_flat.t(), 
                            self.D_i[0], self.t_min_prev, self.t_min, self.t_max, 
                            self.robustness_params)
            logging.info(f"After call_spiking shape: {ti.shape}")
            
            # 重塑输出
            ti = ti.reshape(batch_size, out_height * out_width, self.filters)
            logging.info(f"After reshape shape: {ti.shape}")
            
            ti = ti.permute(0, 2, 1).contiguous()
            ti = ti.view(batch_size, self.filters, out_height, out_width)
            logging.info(f"Final output shape: {ti.shape}")
            
        return ti

    def build(self, input_channels):
        self.kernel = nn.Parameter(
            torch.empty(self.filters, input_channels, *self.kernel_size, dtype=torch.float64))
        
        # 添加默认的初始化方法
        if self.kernel_initializer is None:
            nn.init.xavier_uniform_(self.kernel)
        elif isinstance(self.kernel_initializer, str):
            if self.kernel_initializer == 'glorot_uniform':
                nn.init.xavier_uniform_(self.kernel)
            elif self.kernel_initializer == 'glorot_normal':
                nn.init.xavier_normal_(self.kernel)
            elif self.kernel_initializer == 'he_uniform':
                nn.init.kaiming_uniform_(self.kernel)
            elif self.kernel_initializer == 'he_normal':
                nn.init.kaiming_normal_(self.kernel)
        else:
            self.kernel_initializer(self.kernel)

    def set_params(self, t_min_prev: float, t_min: float):
        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64)
        self.t_min = torch.tensor(t_min, dtype=torch.float64)
        self.t_max = torch.tensor(t_min + self.B_n, dtype=torch.float64)
        return t_min, t_min + self.B_n


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


def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    if robustness_params.get('time_bits', 0) != 0:
        tj = quantize_tensor(tj - t_min_prev, t_min_prev, t_min, 
                           robustness_params['time_bits']) + t_min_prev
    
    if robustness_params.get('weight_bits', 0) != 0:
        W = quantize_tensor(W, robustness_params['w_min'], 
                          robustness_params['w_max'], 
                          robustness_params['weight_bits'])

    threshold = t_max - t_min - D_i
    ti = (torch.matmul(tj - t_min, W) + threshold + t_min)
    ti = torch.where(ti < t_max, ti, t_max)
    
    if robustness_params.get('noise', 0) != 0:
        ti = ti + torch.randn_like(ti) * robustness_params['noise']
    
    return ti


def quantize_tensor(x, min_val, max_val, num_bits):
    # Simple linear quantization
    scale = (2**num_bits - 1) / (max_val - min_val)
    x_scaled = ((x - min_val) * scale).round() / scale + min_val
    return x_scaled

logging.basicConfig(level=logging.INFO)

def test_snn_model():
    try:
        # 创建模型
        model = SNNModel()
        
        # 添加层
        input_dim = 784  # 例如MNIST数据集的输入维度
        hidden_units = 128
        output_units = 10
        
        # 添加一个SpikingDense隐藏层
        hidden_layer = SpikingDense(
            units=hidden_units,
            name='hidden1',
            X_n=1.0,
            input_dim=input_dim,
            outputLayer=False,
            robustness_params={'time_bits': 8}
        )
        model.layers.append(hidden_layer)
        
        # 添加输出层
        output_layer = SpikingDense(
            units=output_units,
            name='output',
            X_n=1.0,
            input_dim=hidden_units,
            outputLayer=True,
            robustness_params={'time_bits': 8}
        )
        model.layers.append(output_layer)
        
        # 创建测试数据
        batch_size = 32
        x = torch.rand(batch_size, input_dim, dtype=torch.float64)
        
        # 运行前向传播
        logging.info("Running forward pass...")
        output, min_ti = model(x)
        
        # 检查输出
        logging.info(f"Output shape: {output.shape}")
        logging.info(f"Min firing times: {min_ti}")
        
        return True
    
    except Exception as e:
        logging.error(f"Test failed with error: {str(e)}")
        return False

def test_spiking_conv():
    try:
        # 创建更小的测试用例
        conv_layer = SpikingConv2D(
            filters=16,
            name='conv1',
            X_n=1.0,
            kernel_size=(3, 3),
            robustness_params={'time_bits': 8, 'weight_bits': 8, 'w_min': -1, 'w_max': 1, 'noise': 0},
            kernel_initializer='glorot_uniform'
        )
        
        # 使用更小的输入尺寸进行测试
        batch_size = 2
        input_channels = 3
        height = 8
        width = 8
        x = torch.rand(batch_size, input_channels, height, width, dtype=torch.float64)
        
        logging.info(f"Test input shape: {x.shape}")
        
        # 构建层
        conv_layer.build(input_channels)
        conv_layer.set_params(0.0, 1.0)
        
        # 计算期望的输出尺寸
        if conv_layer.padding == 'same':
            expected_height = height
            expected_width = width
        else:
            expected_height = height - conv_layer.kernel_size[0] + 1
            expected_width = width - conv_layer.kernel_size[1] + 1
            
        expected_shape = (batch_size, conv_layer.filters, expected_height, expected_width)
        logging.info(f"Expected output shape: {expected_shape}")
        
        # 运行前向传播
        output = conv_layer(x)
        logging.info(f"Actual output shape: {output.shape}")
        
        assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"
        return True
        
    except Exception as e:
        logging.error(f"Convolution test failed with error: {str(e)}")
        logging.error(f"Error location: {e.__traceback__.tb_lineno}")
        return False

if __name__ == "__main__":
    logging.info("Starting tests...")
    
    logging.info("Testing Dense layers...")
    dense_result = test_snn_model()
    logging.info(f"Dense layer test {'passed' if dense_result else 'failed'}")
    
    logging.info("\nTesting Convolution layer...")
    conv_result = test_spiking_conv()
    logging.info(f"Convolution layer test {'passed' if conv_result else 'failed'}")