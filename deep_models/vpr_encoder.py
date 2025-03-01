from utils import recalltw
import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import neuron, surrogate

class EventPoolingEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size:int):
        super().__init__()
        self.conv = nn.Sequential(
                    # For now the kernel is squared otherwise the kernel size should be something like (h,w,d)
                    nn.Conv3d(in_channels, in_channels, kernel_size=(5, kernel_size, kernel_size), stride=1, padding=(0, kernel_size//2, kernel_size//2)),
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0),
                    neuron.IFNode(surrogate_function=surrogate.ATan()),
                )
    def forward(self, x):
        return self.conv(x)

class EventVPREncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size:int, num_places:int, n_hist:int):
        super().__init__()
        self.bottom = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = (5, 7, 7), stride=1, padding=(0,3,3), bias=False, padding_mode='replicate'),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv1 = EventPoolingEncoder(in_channels, out_channels, kernel_size)
        self.conv2 = EventPoolingEncoder(out_channels, out_channels * 2, kernel_size)
        self.conv3 = EventPoolingEncoder(out_channels * 2, out_channels * 3, kernel_size)
        self.conv4 = EventPoolingEncoder(out_channels * 3, out_channels * 4, kernel_size)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            # TODO: add a dropout layer with a variable dropout rate
            # nn.Dropout(p=0.5),
            nn.Linear(out_channels * 4 * (n_hist-4*4) * 16 * 21, 128), # [BatchS, Outchannel (128), 3, 16, 21]
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(128, num_places)
        )

    def get_n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_parameters(self, layer):
        return sum(p.numel() for p in layer.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.decoder(x)
        return x
    
class EmbeddedVPREncoder(EventVPREncoder):
    def __init__(self, in_channels: int, out_channels: int, kernel_size:int, num_places:int, embedding_size:int, n_hist:int):
        super().__init__(in_channels,out_channels, kernel_size, num_places, n_hist)

        # Override decoder with a readout of the dimension of the embedding for constrastive loss
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 4 * (n_hist-4*4) * 16 * 21, embedding_size), # [BatchS, Outchannel (128), 3, 16, 21]
        )
    
def convert_hist_tensor(batch_size:int, hists:np.array, dims:tuple, mode="3d") -> torch.tensor:
    '''
    Convert histogram in tensor.
    Args:
        batch_size (int): The number of samples in a batch.
        hists (list): A list of histograms, where each histogram is a 2D array.
        dims (tuple): The dimensions of each histogram in the format (height, width).
        mode (str): The mode of conversion. Can be either '2d' or '3d'.
                    If '2d', returns a tensor of shape [B, C, H, W].
                    If '3d', returns a tensor of shape [B, C, N_HIST, H, W].

    Returns:
        torch.Tensor: A tensor of shape [B, C, N_HIST, W, H] where:
            B is the batch size,
            C is the number of channels (2 for ON and OFF events),
            N_HIST is the number of histograms (len(hists)),
            W is the width of the histogram,
            H is the height of the histogram.
    '''
    hist_tensor = torch.zeros(batch_size, 2, len(hists), dims[0], dims[1])
    if mode == "2d": # 2D Case
        # hist_tensor = torch.zeros(batch_size, 2, dims[0], dims[1])
        # for hist in hists:
        #     on_events = torch.from_numpy(hist[:, :, 0].astype(np.float32))  # ON events
        #     off_events = torch.from_numpy(hist[:, :, 1].astype(np.float32))  # OFF events
        #     hist_tensor[:, 0, :, :] = on_events
        #     hist_tensor[:, 1, :, :] = off_events
        # return hist_tensor
        hist_tensor = torch.zeros(batch_size, len(hists) * 2, dims[0], dims[1])
        for idx, hist in enumerate(hists):
            on_events = torch.from_numpy(hist[:, :, 0].astype(np.float32))  # ON events
            off_events = torch.from_numpy(hist[:, :, 1].astype(np.float32))  # OFF events
            hist_tensor[:, idx+idx, :, :] = on_events
            hist_tensor[:, idx+idx+1, :, :] = off_events
        return hist_tensor
    elif mode == "3d":
        for i, hist in enumerate(hists): # 3D Case
            hist_tensor[:, 0, i, :, :] = torch.from_numpy(hist[:, :, 0])  # ON events
            hist_tensor[:, 1, i, :, :] = torch.from_numpy(hist[:, :, 1])  # OFF events
        return hist_tensor
    else:
        raise ValueError(f"Invalid mode: {mode}. Please choose either '2d' or '3d'.")

if __name__ == "__main__":
    net = EventVPREncoder(in_channels=2, out_channels=32, kernel_size=7, num_places=5)

    event_seq = recalltw.get_event_seq("sunset1", 25, 0.06)
    event_seq = recalltw.time_windows_around(event_seq[0],0.06,20)
    in_tensor = convert_hist_tensor(1,event_seq, [260,346])
    net(in_tensor)
    print(f"Total parameters: {net.get_n_parameters()}")
    print(f"Parameters in Conv1: {net.get_layer_parameters(net.conv1)}")
    print(f"Parameters in Conv2: {net.get_layer_parameters(net.conv2)}")
    print(f"Parameters in Conv3: {net.get_layer_parameters(net.conv3)}")
    print(f"Parameters in Conv3: {net.get_layer_parameters(net.conv4)}")
    print(f"Parameters in Bottom: {net.get_layer_parameters(net.bottom)}")
    print(f"Parameters in Linear Layers: {net.get_layer_parameters(net.decoder)}")
