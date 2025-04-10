import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

        if grad_type == 'linear':
            grad = F.relu(1 - input.abs())
        elif grad_type == 'rectangular':
            grad = (input.abs() < 0.5).float()
        elif grad_type == 'triangular':
            sharpness = 1
            grad = (1 / (sharpness ** 2)) * torch.clamp(sharpness - torch.abs(input), min=0)
        elif grad_type == 'gaussian':
            grad = gaussian(input, mu=0.0, sigma=lens)
        elif grad_type == 'multi_gaussian':
            grad = gaussian(input, mu=0.0, sigma=lens) * (1. + hight) \
                   - gaussian(input, mu=lens, sigma=6 * lens) * hight \
                   - gaussian(input, mu=-lens, sigma=6 * lens) * hight
        elif grad_type == 'slayer':
            grad = torch.exp(-5 * input.abs())
        else:
            raise ValueError(f"Unsupported gradient type: {grad_type}")

        return grad_input * grad * gamma, None, None, None, None


class LIFRateNeuronLayer(nn.Module):
    def __init__(self, num_neurons, thresh=1.0, tau_mem=20.0, tau_syn=10.0,
                 dt=1.0, reset_mode='soft', grad_type='triangular',
                 lens=0.5, gamma=1.0, hight=0.15, learn_params=True):
        super().__init__()
        self.num_neurons = num_neurons
        self.lens = lens
        self.gamma = gamma
        self.hight = hight

        if learn_params:
            self.dt = nn.Parameter(torch.tensor(dt, dtype=torch.float32))
            self.thresh = nn.Parameter(torch.tensor([thresh], dtype=torch.float32))
            self.tau_mem = nn.Parameter(torch.tensor([tau_mem], dtype=torch.float32))
            self.tau_syn = nn.Parameter(torch.tensor([tau_syn], dtype=torch.float32))
        else:
            self.register_buffer('dt', torch.tensor(dt, dtype=torch.float32))
            self.register_buffer('thresh', torch.tensor([thresh], dtype=torch.float32))
            self.register_buffer('tau_mem', torch.tensor([tau_mem], dtype=torch.float32))
            self.register_buffer('tau_syn', torch.tensor([tau_syn], dtype=torch.float32))

        self.reset_mode = reset_mode
        self.grad_type = grad_type

        # 添加状态变量
        self.syn = None
        self.mem = None

    def forward(self, x):
        """
        单个时间步的前向传播
        x: [batch_size, num_neurons]
        """
        # 计算常数
        alpha_syn = torch.exp(-self.dt / self.tau_syn)
        alpha_mem = torch.exp(-self.dt / self.tau_mem)
        input_scale = (1 - alpha_syn) * self.tau_syn

        # 更新突触和膜电位状态
        self.syn = alpha_syn * self.syn + input_scale * x
        self.mem = alpha_mem * self.mem + (1 - alpha_mem) * self.syn

        # 生成脉冲
        spike = SurrogateGradientFunction.apply(
            self.mem - self.thresh,
            self.grad_type,
            self.lens,
            self.gamma,
            self.hight
        )

        # 重置膜电位
        if self.reset_mode == 'soft':
            self.mem = self.mem - spike * self.thresh
        else:
            self.mem = self.mem * (1 - spike)

        return spike, self.mem

    def set_neuron_state(self, batch_size, device):
        """初始化或重置神经元状态"""
        self.syn = torch.zeros(batch_size, self.num_neurons, device=device)
        self.mem = torch.zeros(batch_size, self.num_neurons, device=device)


class SNNRateLayer(nn.Module):
    def __init__(self, in_features, out_features, **neuron_params):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.neurons = LIFRateNeuronLayer(num_neurons=out_features, **neuron_params)

    def forward(self, x):
        """
        单个时间步的前向传播
        x: [batch_size, in_features]
        """
        # 计算带权重的输入
        weighted = torch.matmul(x, self.weight)  # [batch_size, out_features]

        # 通过LIF神经元处理
        spike, mem = self.neurons(weighted)

        return spike, mem

    def set_neuron_state(self, batch_size, device):
        """初始化或重置层的神经元状态"""
        self.neurons.set_neuron_state(batch_size, device)
