import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, surrogate

class EmbeddingEventNet(nn.Module):
    # TODO Should change MultiplyBy to BatchNorm (and add learnable parameters)
    def __init__(self, channels=2, multiply_factor=5., step_mode='m'):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 32, 5),
            MultiplyBy(multiply_factor),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5),
            MultiplyBy(multiply_factor),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 5),
            MultiplyBy(multiply_factor),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            MultiplyBy(multiply_factor),
            neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=float('inf'), v_reset=0.)
        )
        
    def forward(self, x):
        x = self.convnet(x)
        x = self.fc(x)
        return x

    def get_avg_spike_rate(self, x):
        firing_rates_dict = {
            "out_conv1": 0,
            "out_conv2": 0,
            "out_conv3": 0,
            "out_fc": 0,
        }

        keys = list(firing_rates_dict.keys())

        for module in self.convnet:
            if isinstance(module, neuron.IFNode):
                x = module(x)
                firing_rates_dict[keys[0]] = x.count_nonzero() / x.numel()
                keys.pop(0)
            else:
                x = module(x)
        return firing_rates_dict

    def get_last_layer_potential(self, x):
        x = self.forward(x)
        return x[-1].v


class EmbeddingEventNetL2(EmbeddingEventNet):
    def __init__(self, channels=2, multiply_factor=5., step_mode='m'):
        super().__init__(channels=channels, multiply_factor=multiply_factor, step_mode=step_mode)
    def forward(self, x):
        x = super(EmbeddingEventNetL2, self).forward(x)
        return F.normalize(x, p=2, dim=1)

class MultiplyBy(nn.Module):
    '''
    From Rancon et al. 2022, Stereospike (https://github.com/urancon/StereoSpike/blob/main/network/blocks.py)
    This layer multiplies the input by a learnable or fixed scale value to ensure spiking neurons
    to actually spike and solve the vanishing gradient problem without BatchNorm.
    '''
    def __init__(self, scale_value: float = 5., learnable: bool = False):
        super(MultiplyBy, self).__init__()

        if learnable:
            self.scale_value = nn.Parameter(torch.tensor(scale_value))
        else:
            self.scale_value = scale_value

    def forward(self, x):
        return x * self.scale_value
    
def init_weights(network):
    # Recursive weight initialization
    for module in network.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)