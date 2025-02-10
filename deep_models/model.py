import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate

class SNNEncoderStateless(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SNNEncoderStateless, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        self.fc = nn.Linear(86016, output_dim)
        self.sn_final = neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=float("inf"), v_reset=0.)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sn_final(x)
        x = self.sn_final.v.detach()
        return x

class SNNEncoderStateful(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SNNEncoderStateful, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # neuron.ParametricLIFNode(surrogate_function=surrogate.ATan())
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # neuron.ParametricLIFNode(surrogate_function=surrogate.ATan())
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #neuron.ParametricLIFNode(surrogate_function=surrogate.ATan())
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #neuron.ParametricLIFNode(surrogate_function=surrogate.ATan())
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )
        self.fc = nn.Linear(86016, output_dim)
        # self.sn_final = neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), v_threshold=float("inf"), v_reset=0.)
        self.sn_final = neuron.LIFNode(surrogate_function=surrogate.ATan(), v_threshold=float("inf"), v_reset=0.)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sn_final(x)
        x = self.sn_final.v.detach()
        return x

    def get_parametric_lif_tau(self):
        tau_values = []
        for module in self.modules():
            if isinstance(module, neuron.ParametricLIFNode): # Tau is not stored directly, we need to calculate it though the weight
                plif_w = module.w
                tau = 1 / plif_w.sigmoid()
                tau_values.append(tau.item())
        return tau_values

