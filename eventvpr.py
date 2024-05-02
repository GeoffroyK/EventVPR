import torch
import torch.nn as nn

from torch.utils.data import Dataset
from spikingjelly.activation_based import neuron, learning, functional

class EventVPR(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, n_in, n_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Input shape should be [B, C, W, H] with C = 2, ON & OFF Channel
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, n_out, bias=False),
            neuron.LIFNode(tau=2.)
        )

    def forward(self, x):
        return self.net(x)
    
class eventDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, eventsPath, spikeNumber) -> None:
        super().__init__()
        self.eventPath = eventsPath
        self.spikeNumber = spikeNumber

        self.imgList, self.timestamps = self.read_img_file()

        self.tbins = []
        self.sbins = self.buildSpikeBins()


    def buildTimeBins(self) -> None:
        pass

    def buildSpikeBins(self) -> list:
        '''
        Divide the dataset, in N bins composed of spikesNumber spike(s).
        If spikesNumber = 1, then one spike will only be taken during each bins (eg, no binnings) 
        '''
        spikeBins = []
        # Iterate over the events
        
        spikeIndex = 0
        currentBins = []
        for index, event in enumerate(timestamps):
            if spikeIndex == self.spikeNumber:
                spikeIndex = 0
                spikeBins.append(currentBins)
                currentBins = []

            # Update last bins if the list is finished
            if index == len(timestamps):
                spikeBins.append(currentBins)

            currentBins.append(event)

        return spikeBins


    def read_img_file(self):
        img_list = []
        timestamps = []
        with open(self.eventPath + "images.txt") as file:
            for item in file:
                timestamps.append(float(item.split(' ')[0].strip()))
                img_list.append(self.eventPath + item.split(' ')[1].strip())
        assert len(img_list) == len(timestamps)
        return img_list, timestamps
    
    def __len__(self) -> int:
        return len(self.tins) + len(self.sbins)

    def __getitem__(self, index) -> any:
        return super().__getitem__(index)

    def spikeBinning():
        """_summary_
        """
        pass

    def timeBinning():
        """_summary_
        """
        pass

if __name__=="__main__":
    net = EventVPR(n_in=1,n_out=2)
    dataset = eventDataset(eventsPath='/home/keimeg/Téléchargements/shapes_rotation/', spikeNumber=1000)
    imglist, timestamps = dataset.read_img_file()