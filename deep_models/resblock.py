import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron

class Spiking_MLP(nn.Module):
    '''
    A multi-layer perceptron (MLP) for spiking neural networks, consisting of two linear layers with LIF neuron activation.
    
    Attributes:
        tau (float): The time constant of the LIF neuron.
        dim (tuple): The dimensions of the input and output tensors.
    '''
    def __init__(self, tau:float=2.0, dim:tuple=(346,260)):
        super(Spiking_MLP, self).__init__()
        self.tau = tau
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim[0]*dim[1] * 2, out_features=dim[0]*dim[1] * 2),
            neuron.LIFNode(self.tau),
            nn.Linear(in_features=dim[0]*dim[1] * 2, out_features=dim[0]*dim[1] * 2), 
            neuron.LIFNode(self.tau) # [B,T,C,H,W]
        )
    def forward(self, x):
        return self.mlp(x)

class LIF_ResBlock(nn.Module):
    '''
    A residual block for spiking neural networks, consisting of two convolutional layers with batch normalization and LIF neuron activation.
    
    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int, optional): The size of the kernel. Defaults to 3.
        stride (int, optional): The stride of the convolution. Defaults to 1.
        tau (float, optional): The time constant of the LIF neuron. Defaults to 2.0.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, tau=2.0):
        super(LIF_ResBlock, self).__init__()
        
        # Padding is calculated based on kernel size to keep the spatial dimensions consistent
        padding = (kernel_size - 1) // 2
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # LIF neuron activation layer with specified tau
        self.lif1 = neuron.LIFNode(tau=tau)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Second LIF neuron activation layer with specified tau
        self.lif2 = neuron.LIFNode(tau=tau)
        
        # Downsample layer for the skip connection, if dimensions need to be matched
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.downsample(x)  # Adjust input dimensions if necessary
        
        # First convolution + batch norm + LIF
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)  # Spiking activation with tau
        
        # Second convolution + batch norm + LIF
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lif2(out)  # Spiking activation with tau
        
        # Add skip connection (identity) and apply LIF
        out += identity
        out = self.lif2(out)  # Final spiking activation on the combined output
        
        return out
    

if __name__ == "__main__":
    from utils.data_augmentation import get_event_seq
    event_seq = get_event_seq("sunset1", 1, 1.0, 'txt')
    place = 0
    batch_size = 1
    mlp = Spiking_MLP()

    print("feeding to net")
    n_time = len(event_seq[place])
    in_tensor = torch.zeros(batch_size, n_time, 2, 346, 240)
    
    t_previous = event_seq[0][0]

    for event in event_seq[place]:
        t = event[0]
        x = int(event[1])
        y = int(event[2])
        c = int(event[3])
        mlp(t)
        t = torch.tensor(t)
