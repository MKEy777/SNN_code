import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
from tqdm import tqdm


# 高斯函数和代理梯度
def gaussian(x, mu=0.0, sigma=1.0):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))


class SurrogateGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_type, lens, gamma, hight):
        ctx.save_for_backward(input)
        ctx.grad_type = grad_type
        ctx.lens = lens
        ctx.gamma = gamma
        ctx.hight = hight
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_type = ctx.grad_type
        lens = ctx.lens
        gamma = ctx.gamma
        hight = ctx.hight

        if grad_type == 'gaussian':
            grad = gaussian(input, mu=0.0, sigma=lens)
        else:
            raise ValueError(f"不支持的梯度类型: {grad_type}")

        return grad_input * grad * gamma, None, None, None, None


# LIF 神经元层
class LIFTTFSNeuronLayer(nn.Module):
    def __init__(self, num_neurons, thresh=1.0, dt=1.0, reset_mode='soft',
                 grad_type='gaussian', lens=0.5, gamma=1.0, hight=0.15, learn_params=True, is_output_layer=False):
        super().__init__()
        self.num_neurons = num_neurons
        self.is_output_layer = is_output_layer
        if learn_params:
            self.dt = nn.Parameter(torch.tensor(dt))
            self.thresh = nn.Parameter(torch.ones(num_neurons) * thresh)
            self.bias = nn.Parameter(torch.zeros(num_neurons)) if is_output_layer else None
        else:
            self.register_buffer('dt', torch.tensor(dt))
            self.register_buffer('thresh', torch.ones(num_neurons) * thresh)
            self.bias = torch.zeros(num_neurons) if is_output_layer else None
        self.reset_mode = reset_mode
        self.grad_type = grad_type
        self.lens = lens
        self.gamma = gamma
        self.hight = hight
        self.syn = None
        self.mem = None
        self.has_spiked = None
        self.spike_time = None
        self.t_min = 1.0
        self.t_min_prev = 0.0

    def forward(self, x, current_time, input_spike_time=None):
        if self.is_output_layer:
            if input_spike_time is None:
                raise ValueError("Output layer requires input spike times")
            time_diff = self.t_min - input_spike_time
            output = torch.matmul(time_diff, x)
            output = output + self.bias
            return output
        else:
            alpha = torch.exp(-self.dt)
            mask = (~self.has_spiked).float()
            self.mem = alpha * self.mem + (1 - alpha) * x * mask
            spike = SurrogateGradientFunction.apply(
                self.mem - self.thresh, self.grad_type, self.lens, self.gamma, self.hight)
            new_spikes = (spike > 0) & (~self.has_spiked)
            self.spike_time = torch.where(new_spikes, current_time, self.spike_time)
            self.has_spiked = self.has_spiked | (spike > 0)
            if self.reset_mode == 'soft':
                self.mem = self.mem - spike * self.thresh * mask
            else:
                self.mem = self.mem * (1 - spike) * mask
            return spike

    def set_neuron_state(self, batch_size, device, default_spike_time=float('inf')):
        self.syn = torch.zeros(batch_size, self.num_neurons, device=device)
        self.mem = torch.zeros(batch_size, self.num_neurons, device=device)
        self.has_spiked = torch.zeros(batch_size, self.num_neurons, dtype=torch.bool, device=device)
        self.spike_time = torch.full((batch_size, self.num_neurons), default_spike_time, device=device)


# SNN 层
class SNNTTFSLayer(nn.Module):
    def __init__(self, in_features, out_features, is_output_layer=False, **neuron_params):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.normal_(self.weight, mean=0, std=0.1)
        self.neurons = LIFTTFSNeuronLayer(num_neurons=out_features, is_output_layer=is_output_layer, **neuron_params)
        self.is_output_layer = is_output_layer

    def forward(self, x, current_time, input_spike_time=None):
        if self.is_output_layer:
            return self.neurons(self.weight, current_time, input_spike_time)
        else:
            weighted = torch.matmul(x, self.weight)
            return self.neurons(weighted, current_time)

    def set_neuron_state(self, batch_size, device, default_spike_time=float('inf')):
        self.neurons.set_neuron_state(batch_size, device, default_spike_time)