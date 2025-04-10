import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from utils.load_dataset_deap import dataset_prepare

from module.LIF import *


class TwoLayerSNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=200, output_dim=2):
        super(TwoLayerSNN, self).__init__()

        # 第一层参数
        self.layer1_params = {
            'thresh': 0.5,
            'tau_mem': 10.0,
            'tau_syn': 5.0,
            'dt': 1.0,
            'reset_mode': 'soft',
            'grad_type': 'gaussian',
            'lens': 0.5,
            'gamma': 1.0,
            'hight': 0.15,
            'learn_params': True
        }

        # 第二层参数
        self.layer2_params = {
            'thresh': 1.0,
            'tau_mem': 20.0,
            'tau_syn': 10.0,
            'dt': 1.0,
            'reset_mode': 'soft',
            'grad_type': 'multi_gaussian',
            'lens': 0.4,
            'gamma': 1.2,
            'hight': 0.2,
            'learn_params': True
        }

        # 创建两层网络
        self.dense_1 = SNNRateLayer(input_dim, hidden_dim, **self.layer1_params)
        self.dense_2 = SNNRateLayer(hidden_dim, output_dim, **self.layer2_params)

    def set_neuron_state(self, batch_size, device):
        # 重置第一层状态
        self.dense_1.neurons.syn = torch.zeros(batch_size, self.dense_1.neurons.num_neurons, device=device)
        self.dense_1.neurons.mem = torch.zeros(batch_size, self.dense_1.neurons.num_neurons, device=device)

        # 重置第二层状态
        self.dense_2.neurons.syn = torch.zeros(batch_size, self.dense_2.neurons.num_neurons, device=device)
        self.dense_2.neurons.mem = torch.zeros(batch_size, self.dense_2.neurons.num_neurons, device=device)

    def forward(self, input):
        # input shape: [batch_size, seq_length, input_dim]
        batch_size, seq_length, input_dim = input.shape
        device = input.device

        # 初始化神经元状态
        self.set_neuron_state(batch_size, device)

        # 存储输出
        output = torch.zeros(batch_size, self.dense_2.neurons.num_neurons, device=device)

        # 存储每个时间步的状态（如果需要）
        all_spikes_1 = []
        all_mems_1 = []
        all_spikes_2 = []
        all_mems_2 = []

        # 按时间步处理
        for t in range(seq_length):
            # 获取当前时间步的输入
            input_t = input[:, t, :]

            # 第一层前向传播
            weighted_1 = torch.matmul(input_t, self.dense_1.weight)

            # 更新第一层状态
            alpha_syn_1 = torch.exp(-self.dense_1.neurons.dt / self.dense_1.neurons.tau_syn)
            alpha_mem_1 = torch.exp(-self.dense_1.neurons.dt / self.dense_1.neurons.tau_mem)
            input_scale_1 = (1 - alpha_syn_1) * self.dense_1.neurons.tau_syn

            self.dense_1.neurons.syn = alpha_syn_1 * self.dense_1.neurons.syn + input_scale_1 * weighted_1
            self.dense_1.neurons.mem = alpha_mem_1 * self.dense_1.neurons.mem + \
                                       (1 - alpha_mem_1) * self.dense_1.neurons.syn

            # 生成第一层脉冲
            spike_1 = SurrogateGradientFunction.apply(
                self.dense_1.neurons.mem - self.dense_1.neurons.thresh,
                self.dense_1.neurons.grad_type,
                self.dense_1.neurons.lens,
                self.dense_1.neurons.gamma,
                self.dense_1.neurons.hight
            )

            # 重置第一层膜电位
            if self.dense_1.neurons.reset_mode == 'soft':
                self.dense_1.neurons.mem = self.dense_1.neurons.mem - spike_1 * self.dense_1.neurons.thresh
            else:
                self.dense_1.neurons.mem = self.dense_1.neurons.mem * (1 - spike_1)

            # 第二层前向传播
            weighted_2 = torch.matmul(spike_1, self.dense_2.weight)

            # 更新第二层状态
            alpha_syn_2 = torch.exp(-self.dense_2.neurons.dt / self.dense_2.neurons.tau_syn)
            alpha_mem_2 = torch.exp(-self.dense_2.neurons.dt / self.dense_2.neurons.tau_mem)
            input_scale_2 = (1 - alpha_syn_2) * self.dense_2.neurons.tau_syn

            self.dense_2.neurons.syn = alpha_syn_2 * self.dense_2.neurons.syn + input_scale_2 * weighted_2
            self.dense_2.neurons.mem = alpha_mem_2 * self.dense_2.neurons.mem + \
                                       (1 - alpha_mem_2) * self.dense_2.neurons.syn

            # 生成第二层脉冲
            spike_2 = SurrogateGradientFunction.apply(
                self.dense_2.neurons.mem - self.dense_2.neurons.thresh,
                self.dense_2.neurons.grad_type,
                self.dense_2.neurons.lens,
                self.dense_2.neurons.gamma,
                self.dense_2.neurons.hight
            )

            # 重置第二层膜电位
            if self.dense_2.neurons.reset_mode == 'soft':
                self.dense_2.neurons.mem = self.dense_2.neurons.mem - spike_2 * self.dense_2.neurons.thresh
            else:
                self.dense_2.neurons.mem = self.dense_2.neurons.mem * (1 - spike_2)

            # 累积输出
            if t > 0:
                output += self.dense_2.neurons.mem

            # 存储状态（如果需要）
            all_spikes_1.append(spike_1)
            all_mems_1.append(self.dense_1.neurons.mem)
            all_spikes_2.append(spike_2)
            all_mems_2.append(self.dense_2.neurons.mem)

        # 计算平均输出
        output = output / seq_length

        # 如果需要返回所有状态
        all_states = {
            'spikes_1': torch.stack(all_spikes_1, dim=1),  # [batch_size, seq_length, hidden_dim]
            'mems_1': torch.stack(all_mems_1, dim=1),
            'spikes_2': torch.stack(all_spikes_2, dim=1),  # [batch_size, seq_length, output_dim]
            'mems_2': torch.stack(all_mems_2, dim=1)
        }

        return output, all_states


# 使用示例
input_dim = 32
hidden_dim = 200
output_dim = 2
batch_size = 16
seq_length = 100

# 创建模型
model = TwoLayerSNN(input_dim, hidden_dim, output_dim)

# 创建输入数据
input_data = torch.randn(batch_size, seq_length, input_dim)

# 前向传播
output, all_states = model(input_data)

print("Output shape:", output.shape)  # [batch_size, output_dim]
print("Layer 1 spikes shape:", all_states['spikes_1'].shape)  # [batch_size, seq_length, hidden_dim]
print("Layer 2 spikes shape:", all_states['spikes_2'].shape)  # [batch_size, seq_length, output_dim]



