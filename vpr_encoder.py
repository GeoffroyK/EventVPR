import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
import recalltw

class EventVPREncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size:int):
        super().__init__()
        self.conv1 = nn.Sequential(
            # For now the kernel is squared otherwise the kernel size should be something like (h,w,d)
            nn.Conv3d(in_channels, out_channels, kernel_size=(5, kernel_size, kernel_size), stride=1, padding= (0, kernel_size//2, kernel_size//2)),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=0),
            neuron.IFNode(),
            nn.Flatten(),
            nn.Linear(out_channels * 120 * 173, 25),
            neuron.IFNode()
        )

    def forward(self, x):
        return self.conv1(x)
    
def convert_hist_tensor(batch_size:int, hists:list, dims:tuple) -> torch.tensor:
    '''
    Convert histogram in tensor.
    Args:
        batch_size (int): The number of samples in a batch.
        hists (list): A list of histograms, where each histogram is a 2D array.
        dims (tuple): The dimensions of each histogram in the format (height, width).

    Returns:
        torch.Tensor: A tensor of shape [B, C, 1, W, H] where:
            B is the batch size,
            C is the number of channels (2 for ON and OFF events),
            1 is a singleton dimension,
            W is the width of the histogram,
            H is the height of the histogram.
    '''
    hist_tensor = torch.zeros(batch_size, 2, len(hists), dims[0], dims[1])
    for i, hist in enumerate(hists):
        hist_tensor[:, 0, i, :, :] = torch.from_numpy(hist[:, :, 0])  # ON events
        hist_tensor[:, 1, i, :, :] = torch.from_numpy(hist[:, :, 1])  # OFF events
    return hist_tensor



if __name__ == "__main__":
    net = EventVPREncoder(in_channels=2, out_channels=32, kernel_size=7)
    in_tensor = torch.randn(1, 2, 21, 240, 346) # Shape = [B, C, N_HIST, W, H]
    event_seq = recalltw.get_event_seq("sunset1", 25, 0.06)
    event_seq = recalltw.event_histogram(event_seq[0])

    print(convert_hist_tensor(1,[event_seq], [260,346]))


