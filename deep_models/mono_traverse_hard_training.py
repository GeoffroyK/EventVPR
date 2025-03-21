import torch 
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import random
from spikingjelly.clock_driven import neuron, surrogate, functional
from utils.recalltw import time_windows_around

from vpr_encoder import convert_hist_tensor
from constant_paths import hot_pixels_locations


# Seeding for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if using multi-GPU
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SNNEncoder, self).__init__()
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
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(236),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        # 176128 full resolution
        # 131328 cropping 20, 20
        # 155648 cropping 0, 20
        # 135168 cropping 0, 40 
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

class MonoTraverseDataset(torch.utils.data.Dataset):
    def __init__(self, traverses, n_places, time_window, n_hist, format, mode='3d'):
        self.mode = mode
        self.data = []
        self.labels = []
        self.traverses = traverses
        self.n_places = n_places
        self.time_window = time_window
        self.n_hist = n_hist
        self.format = format

        for traverse in traverses:
            print(f"Processing traverse {traverse}")
            from utils.recalltw import get_event_seq
            
            event_seq = self.get_event_seq_offset(traverse, n_places, 1.0, format, offset=2)
            # print(f"Event sequence shape offset: {event_seq[0].shape}")
            # event_seq = get_event_seq(traverse, n_places, 1.0, format)
            # print(f"Event sequence shape normal: {event_seq[0].shape}")

            for idx, place in enumerate(event_seq):
                event_seq[idx] = self.filter_hot_pixels(traverse, place)

            for place in range(n_places): 
                hist_seq = time_windows_around(event_seq[place], time_window, n_hist)
                tensor = convert_hist_tensor(1, hist_seq, [260, 346], mode=self.mode)
                self.data.append(tensor.squeeze(0))  # Remove batch dimension
                self.labels.append(place)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
   
    def filter_hot_pixels(self, traverse, events, hot_pixels=hot_pixels_locations):
        '''
        Filter out hot pixels from the input tensor.
        Hot pixel are detected though this module ()
        and are caused by the 
        '''
        filepath = hot_pixels[traverse]
        # Read and print first line of hot pixels file
        hot_pixels_coords = np.loadtxt(filepath, delimiter=',', dtype=int)
        keep_mask = ~np.any(np.all(events[:, [1,2]] == hot_pixels_coords[:, None], axis=2), axis=0)
        events = events[keep_mask]
        return events   
    
    def get_event_seq_offset(self, traverse, n_places, timewindow, format='pickle', offset=0) -> list:
        '''
        Get the event sequence of each places on selectected traverse at a fixed timewindow
        with an offset between places
        '''
        curr_offset = 0
        event_seq = []
        for place in range(n_places):
            if format == 'pickle':
                events = np.load(f"/home/geoffroy/Documents/EventVPR/notebooks/extracted_places/{traverse}_{place + curr_offset}_{timewindow}.npy", allow_pickle=True)
            elif format == 'txt':
                events = np.loadtxt(f"/home/geoffroy/Documents/EventVPR/notebooks/extracted_places/{traverse}_{place + curr_offset}_{timewindow}.txt")
            bin, n, e, = events.shape
            events = events.reshape(bin*n, e)    
            event_seq.append(events)
            curr_offset += offset
        # Return list of events instead of concatenating
        
        return event_seq

if __name__ == "__main__":
    train_dataset = MonoTraverseDataset(traverses=["sunset1"], n_places=50, time_window=0.3, n_hist=25, format='pickle', mode='2d')
    val_dataset = MonoTraverseDataset(traverses=["sunset2"], n_places=50, time_window=0.3, n_hist=25, format='pickle', mode='2d')
    from triplet_mining import ProximityBasedHardMiningTripletLoss
    from torch.utils.data import DataLoader
    criterion = ProximityBasedHardMiningTripletLoss(margin=1, distance_threshold=3)
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=25, shuffle=True)
    net = SNNEncoder(input_dim=2*25 + 2, hidden_dim=128, output_dim=128)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(29):
        train_loss = 0
        val_loss = 0
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            net(data)
            output = net.sn_final.v
            functional.reset_net(net)

            loss, train_acc = criterion(target, output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        net.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                output = net(data)
                functional.reset_net(net)
                loss, val_acc = criterion(target, output)
            val_loss += loss.item()

        print(f"Epoch {epoch} Train Loss {train_loss/len(train_loader)}")
        print(f"Epoch {epoch} Train Accuracy {train_acc/len(train_loader)}")
        print(f"Epoch {epoch} Val Loss {val_loss/len(val_loader)}")
        print(f"Epoch {epoch} Val Accuracy {val_acc/len(val_loader)}")