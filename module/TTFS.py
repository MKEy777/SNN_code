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

class LIFTTFSNeuronLayer(nn.Module):
    def __init__(self, num_neurons, thresh=1.0, dt=1.0, reset_mode='soft',
                 grad_type='triangular', lens=0.5, gamma=1.0, hight=0.15, learn_params=True):
        super().__init__()
        self.num_neurons = num_neurons
        self.lens = lens
        self.gamma = gamma
        self.hight = hight

        if learn_params:
            self.dt = nn.Parameter(torch.tensor(dt, dtype=torch.float32))
            self.thresh = nn.Parameter(torch.tensor([thresh], dtype=torch.float32))
        else:
            self.register_buffer('dt', torch.tensor(dt, dtype=torch.float32))
            self.register_buffer('thresh', torch.tensor([thresh], dtype=torch.float32))

        self.reset_mode = reset_mode
        self.grad_type = grad_type

        self.syn = None
        self.mem = None
        self.has_spiked = None
        self.spike_time = None

    def forward(self, x, current_time):
        alpha = torch.exp(-self.dt)
        input_scale = 1 - alpha
        mask = (~self.has_spiked).float()

        self.syn = alpha * self.syn + input_scale * x * mask
        self.mem = alpha * self.mem + (1 - alpha) * self.syn * mask

        spike = SurrogateGradientFunction.apply(
            self.mem - self.thresh,
            self.grad_type,
            self.lens,
            self.gamma,
            self.hight
        )

        new_spikes = (spike > 0) & (~self.has_spiked)
        self.spike_time = torch.where(new_spikes, current_time, self.spike_time)
        self.has_spiked = self.has_spiked | (spike > 0)

        if self.reset_mode == 'soft':
            self.mem = self.mem - spike * self.thresh * mask
        else:
            self.mem = self.mem * (1 - spike) * mask

        return spike, self.spike_time

    def forward_event_driven(self, x, current_time, last_time=None):
        mask = (~self.has_spiked).float()
        if last_time is not None:
            delta_t = current_time - last_time
            self.syn = (self.syn / 2) + (0.5 * x * delta_t * mask)
            self.mem = (self.mem / 2) + (0.5 * self.syn * delta_t * mask)
        else:
            self.syn = (self.syn / 2) + (0.5 * x * mask)
            self.mem = (self.mem / 2) + (0.5 * self.syn * mask)

        spike = SurrogateGradientFunction.apply(
            self.mem - self.thresh,
            self.grad_type,
            self.lens,
            self.gamma,
            self.hight
        )

        new_spikes = (spike > 0) & (~self.has_spiked)
        self.spike_time = torch.where(new_spikes, current_time, self.spike_time)
        self.has_spiked = self.has_spiked | (spike > 0)

        if self.reset_mode == 'soft':
            self.mem = self.mem - spike * self.thresh * mask
        else:
            self.mem = self.mem * (1 - spike) * mask

        return spike, self.spike_time

    def set_neuron_state(self, batch_size, device, default_spike_time=float('inf')):
        self.syn = torch.zeros(batch_size, self.num_neurons, device=device)
        self.mem = torch.zeros(batch_size, self.num_neurons, device=device)
        self.has_spiked = torch.zeros(batch_size, self.num_neurons, dtype=torch.bool, device=device)
        self.spike_time = torch.full((batch_size, self.num_neurons), default_spike_time, device=device)

class SNNTTFSLayer(nn.Module):
    def __init__(self, in_features, out_features, **neuron_params):
        super().__init__()
        self.weight = nn.Parameter(torch.abs(torch.randn(in_features, out_features)))  # 正值权重
        self.neurons = LIFTTFSNeuronLayer(num_neurons=out_features, **neuron_params)

    def forward(self, x, current_time):
        weighted = torch.matmul(x, self.weight)
        spike, spike_time = self.neurons(weighted, current_time)
        return spike, spike_time

    def forward_event_driven(self, x, current_time, last_time=None):
        weighted = torch.matmul(x, self.weight)
        spike, spike_time = self.neurons.forward_event_driven(weighted, current_time, last_time)
        return spike, spike_time

    def set_neuron_state(self, batch_size, device, default_spike_time=float('inf')):
        self.neurons.set_neuron_state(batch_size, device, default_spike_time)

if __name__ == "__main__":
    batch_size = 2
    in_features = 3
    out_features = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layer = SNNTTFSLayer(
        in_features, out_features,
        thresh=0.5,  # 降低阈值
        dt=1.0,
        reset_mode='soft',
        grad_type='triangular',
        lens=0.5,
        gamma=1.0,
        hight=0.15,
        learn_params=True
    )
    layer.to(device)
    layer.set_neuron_state(batch_size, device)

    x = torch.ones(batch_size, in_features, device=device) * 2.0  # 增大输入
    spike_times = []
    event_times = torch.tensor([0.0, 1.5, 2.0, 3.0, 4.5], device=device)

    last_time = None
    for event_time in event_times:
        spike, spike_time = layer.forward_event_driven(x, event_time, last_time)
        spike_times.append(spike_time.clone())
        print(f"Event time: {event_time}, Spike times:\n", spike_time)
        last_time = event_time

    final_spike_times = spike_times[-1]
    print("Final spike times:\n", final_spike_times)