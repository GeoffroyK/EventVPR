import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class EventVPR(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_in, n_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Input shape should be [B, C, W, H] with C = 2, ON & OFF Channel
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(n_in, n_out, bias=False), neuron.LIFNode(tau=2.0)
        )

    def forward(self, x):
        return self.net(x)


class EventDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, filepath) -> None:

    def __init__(self, eventDataFrame) -> None:
        super().__init__()

        self.eventDataFrame = self.initData(filepath)
        # self.imgList, self.timestamps = self.read_img_file()
        self.eventDataFrame = eventDataFrame
        self.tbins = []
        self.sbins = []

    def initData(self, filepath):
        '''
        Return a formatted dataframe of all events in the 
        timestamp, x, y, polarity manner.
        '''
        eventdf = pd.read_csv(filepath, delimiter=' ', header=3, usecols=[0, 1, 2, 3])
        eventdf = eventdf.set_axis(["timestamp","x","y","polarity"], axis=1)

        # Convert X and Y columns in integer
        eventdf["x"] = eventdf["x"].astype(int)
        print(eventdf.dtypes)

        return eventdf
        # self.sbins = self.buildSpikeBins()

    def buildTimeBins(self, frameTiming) -> None:
        """
        Divide the dataset in N time window of size frameTiming.
        """
        pbar = tqdm(range(frameTiming),
                    desc="Slicing all events...", position=0)
        sIndex = 0

        xMax = self.eventDataFrame["x"].max()
        yMax = self.eventDataFrame["y"].max()

        binned_event_frames = []
        bin_count = 1
        event_in = torch.zeros((2, yMax + 1, xMax + 1))

        while sIndex < len(self.eventDataFrame):
            while bin_count* frameTiming < self.eventDataFrame["timestamp"].loc[sIndex] and sIndex < len(self.eventDataFrame):
                polarity = self.eventDataFrame["polarity"].loc[sIndex]
                yPos = self.eventDataFrame["y"].loc[sIndex]
                xPos = self.eventDataFrame["x"].loc[sIndex]
                event_in[polarity][yMax - yPos][xMax - xPos] += 1
            pbar.update(1)
        pbar.close()

    def buildSpikeBins(self, bins) -> list:
        """
        Divide the dataset, in N bins composed of spikesNumber spike(s).
        If spikesNumber = 1 :
        One spike will only be taken during each bins (eg, no binnings)
        """

        pbar = tqdm(
            range(len(self.eventDataFrame) // bins),
            desc="Slicing all events...",
            position=0,
        )

        xMax = self.eventDataFrame["x"].max()
        yMax = self.eventDataFrame["y"].max()

        binned_event_frames = []
        event_in = torch.zeros((2, yMax + 1, xMax + 1))
        iteration = 1
        sIndex = 0

        while sIndex < len(self.eventDataFrame):
            while sIndex < iteration * bins and sIndex < len(self.eventDataFrame):
                # Which channel ie polarity
                polarity = self.eventDataFrame["polarity"].loc[sIndex]
                yPos = self.eventDataFrame["y"].loc[sIndex]
                xPos = self.eventDataFrame["x"].loc[sIndex]

                event_in[polarity][yMax - yPos][xMax - xPos] += 1
                sIndex += 1
            binned_event_frames.append(event_in)
            iteration += 1
        binned_event_frames = torch.from_numpy(binned_event_frames)

        self.sbins = binned_event_frames
        return self.sbins
            pbar.update(1)
        pbar.close()
        return binned_event_frames

    def read_img_file(self):
        img_list = []
        timestamps = []
        with open(self.eventPath + "images.txt") as file:
            for item in file:
                timestamps.append(float(item.split(" ")[0].strip()))
                img_list.append(self.eventPath + item.split(" ")[1].strip())
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
    #net = EventVPR(n_in=1,n_out=2)
    eventdt = eventDataset(filepath="./jAER-events.txt")
    #dataset = eventDataset(eventsPath='/home/keimeg/Téléchargements/shapes_rotation/', spikeNumber=1000)
    #imglist, timestamps = dataset.read_img_file()

if __name__ == "__main__":
    net = EventVPR(n_in=346 * 260, n_out=2)

    filepath = "/home/geoffroy/jAER-events.txt"
    eventdf = pd.read_csv(filepath, delimiter=" ",
                          header=3, usecols=[0, 1, 2, 3])
    eventdf = eventdf.set_axis(["timestamp", "x", "y", "polarity"], axis=1)

    # Convert X and Y columns in integer
    eventdf["x"] = eventdf["x"].astype(int)
    print(eventdf.head(10))

    dataset = EventDataset(eventdf)
    binned_event_list = dataset.buildSpikeBins(bins=1000)
    net(binned_event_list[0])
