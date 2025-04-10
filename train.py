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

        # ��һ�����
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

        # �ڶ������
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

        # ������������
        self.dense_1 = SNNRateLayer(input_dim, hidden_dim, **self.layer1_params)
        self.dense_2 = SNNRateLayer(hidden_dim, output_dim, **self.layer2_params)

    def set_neuron_state(self, batch_size, device):
        # ���õ�һ��״̬
        self.dense_1.neurons.syn = torch.zeros(batch_size, self.dense_1.neurons.num_neurons, device=device)
        self.dense_1.neurons.mem = torch.zeros(batch_size, self.dense_1.neurons.num_neurons, device=device)

        # ���õڶ���״̬
        self.dense_2.neurons.syn = torch.zeros(batch_size, self.dense_2.neurons.num_neurons, device=device)
        self.dense_2.neurons.mem = torch.zeros(batch_size, self.dense_2.neurons.num_neurons, device=device)

    def forward(self, input):
        # input shape: [batch_size, seq_length, input_dim]
        batch_size, seq_length, input_dim = input.shape
        device = input.device

        # ��ʼ����Ԫ״̬
        self.set_neuron_state(batch_size, device)

        # �洢���
        output = torch.zeros(batch_size, self.dense_2.neurons.num_neurons, device=device)

        # �洢ÿ��ʱ�䲽��״̬�������Ҫ��
        all_spikes_1 = []
        all_mems_1 = []
        all_spikes_2 = []
        all_mems_2 = []

        # ��ʱ�䲽����
        for t in range(seq_length):
            # ��ȡ��ǰʱ�䲽������
            input_t = input[:, t, :]

            # ��һ��ǰ�򴫲�
            weighted_1 = torch.matmul(input_t, self.dense_1.weight)

            # ���µ�һ��״̬
            alpha_syn_1 = torch.exp(-self.dense_1.neurons.dt / self.dense_1.neurons.tau_syn)
            alpha_mem_1 = torch.exp(-self.dense_1.neurons.dt / self.dense_1.neurons.tau_mem)
            input_scale_1 = (1 - alpha_syn_1) * self.dense_1.neurons.tau_syn

            self.dense_1.neurons.syn = alpha_syn_1 * self.dense_1.neurons.syn + input_scale_1 * weighted_1
            self.dense_1.neurons.mem = alpha_mem_1 * self.dense_1.neurons.mem + \
                                       (1 - alpha_mem_1) * self.dense_1.neurons.syn

            # ���ɵ�һ������
            spike_1 = SurrogateGradientFunction.apply(
                self.dense_1.neurons.mem - self.dense_1.neurons.thresh,
                self.dense_1.neurons.grad_type,
                self.dense_1.neurons.lens,
                self.dense_1.neurons.gamma,
                self.dense_1.neurons.hight
            )

            # ���õ�һ��Ĥ��λ
            if self.dense_1.neurons.reset_mode == 'soft':
                self.dense_1.neurons.mem = self.dense_1.neurons.mem - spike_1 * self.dense_1.neurons.thresh
            else:
                self.dense_1.neurons.mem = self.dense_1.neurons.mem * (1 - spike_1)

            # �ڶ���ǰ�򴫲�
            weighted_2 = torch.matmul(spike_1, self.dense_2.weight)

            # ���µڶ���״̬
            alpha_syn_2 = torch.exp(-self.dense_2.neurons.dt / self.dense_2.neurons.tau_syn)
            alpha_mem_2 = torch.exp(-self.dense_2.neurons.dt / self.dense_2.neurons.tau_mem)
            input_scale_2 = (1 - alpha_syn_2) * self.dense_2.neurons.tau_syn

            self.dense_2.neurons.syn = alpha_syn_2 * self.dense_2.neurons.syn + input_scale_2 * weighted_2
            self.dense_2.neurons.mem = alpha_mem_2 * self.dense_2.neurons.mem + \
                                       (1 - alpha_mem_2) * self.dense_2.neurons.syn

            # ���ɵڶ�������
            spike_2 = SurrogateGradientFunction.apply(
                self.dense_2.neurons.mem - self.dense_2.neurons.thresh,
                self.dense_2.neurons.grad_type,
                self.dense_2.neurons.lens,
                self.dense_2.neurons.gamma,
                self.dense_2.neurons.hight
            )

            # ���õڶ���Ĥ��λ
            if self.dense_2.neurons.reset_mode == 'soft':
                self.dense_2.neurons.mem = self.dense_2.neurons.mem - spike_2 * self.dense_2.neurons.thresh
            else:
                self.dense_2.neurons.mem = self.dense_2.neurons.mem * (1 - spike_2)

            # �ۻ����
            if t > 0:
                output += self.dense_2.neurons.mem

            # �洢״̬�������Ҫ��
            all_spikes_1.append(spike_1)
            all_mems_1.append(self.dense_1.neurons.mem)
            all_spikes_2.append(spike_2)
            all_mems_2.append(self.dense_2.neurons.mem)

        # ����ƽ�����
        output = output / seq_length

        # �����Ҫ��������״̬
        all_states = {
            'spikes_1': torch.stack(all_spikes_1, dim=1),  # [batch_size, seq_length, hidden_dim]
            'mems_1': torch.stack(all_mems_1, dim=1),
            'spikes_2': torch.stack(all_spikes_2, dim=1),  # [batch_size, seq_length, output_dim]
            'mems_2': torch.stack(all_mems_2, dim=1)
        }

        return output, all_states


# ʹ��ʾ��
input_dim = 32
hidden_dim = 200
output_dim = 2
batch_size = 16
seq_length = 100

# ����ģ��
model = TwoLayerSNN(input_dim, hidden_dim, output_dim)

# ������������
input_data = torch.randn(batch_size, seq_length, input_dim)

# ǰ�򴫲�
output, all_states = model(input_data)

print("Output shape:", output.shape)  # [batch_size, output_dim]
print("Layer 1 spikes shape:", all_states['spikes_1'].shape)  # [batch_size, seq_length, hidden_dim]
print("Layer 2 spikes shape:", all_states['spikes_2'].shape)  # [batch_size, seq_length, output_dim]



