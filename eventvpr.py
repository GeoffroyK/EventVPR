import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, encoding, learning, functional
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class EventVPR(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_in, n_out, w_mean, epochs, training_data,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Input shape should be [B, C, W, H] with C = 2, ON & OFF Channel
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(n_in, n_out, bias=False), neuron.LIFNode(tau=2.0)
        )

        self.epochs = epochs
        self.w_mean = w_mean
        self.training_data = training_data

    def forward(self, x):
        return self.net(x)

    def stdp_training(self, optimizer, learner) -> nn.Module:
        """ STDP Training with grayscaled, patches images of the nordland dataset

        Args:
            encoder (_type_): _description_
            optimizer (_type_): _description_
            learner (_type_): _description_

        Returns:
            nn.Module: _description_
        """
    
        # Check and define the training device (CUDA GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init tqdm bar for training progress
        pbar = tqdm(total=self.epochs,
            desc=f"Training of model with STDP on {device}",
            position=0)

        # Initial weights of the network on the Linear layers
        for layer in self.net.children():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, mean=self.w_mean) 

        # Define dictionnary for inhibition
        selective_neuron = {}

        # Begin STDP training
        for _ in range(self.epochs):
            self.net.train()

            for inputs in self.training_data:
                optimizer.zero_grad()
                inputs.to(device)

                winner_index = 0
                inhibited_neurons = []

                # Discretization of the input image in spike train using a pixel intensity to latency encoder
                for _ in range(100): #TODO change in variable
                    out_spike = self.net(inputs[0]).detach() #TODO change this to enable batch

                """  # Create inhibition to make a Winner-take-All inhibiton mechanism
                    if 1. in out_spike: # If at least one spike has occured
                        winner_index = torch.argmax(out_spike)
                    if label not in selective_neuron and winner_index not in selective_neuron.values():
                        selective_neuron[label] = winner_index
                    if selective_neuron.__len__() > 0 and label in selective_neuron:
                        for idx in range(len(self.net.layer[-1].v[0].detach())): # Inhib all non-winning spiking neurons of the output layer.
                            if idx != selective_neuron[label]:
                                inhibited_neurons.append(idx) """

                # Prevent non-winning neurons from spiking with a negative fixed potential
                for neuron in inhibited_neurons:
                    self.net.layer[-1].v[0][neuron] = -10 #TODO change this fixed parameter to a variable !

                # Clamp the weights of the network between 0 and 1 to avoid huge values to appear
                self.net[1].weight.data = torch.clamp(self.net[1].weight.data, 0, 1)

                # Calculate the delta of weights (STDP step)
                learner.step(on_grad=True)
                optimizer.step()
                
                # Reset network state between each images to have a fair learning basis for each images
                functional.reset_net(self.net)
                # Reset the stdp learner to avoid memory overflow
                learner.reset()

            pbar.update(1) # Update tqdm bar
        pbar.close() # Close tqdm bar
        return self.net

class EventDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, filepath) -> None:
        super().__init__()

        self.eventDataFrame = self.initData(filepath)

        self.xMax = self.eventDataFrame["x"].max()
        self.yMax = self.eventDataFrame["y"].max()

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

    def buildTimeBins(self, frameTiming) -> None:
        """
        Divide the dataset in N time window of size frameTiming.
        """
        pbar = tqdm(range(frameTiming),
                    desc="Slicing all events...", position=0)
        sIndex = 0

        binned_event_frames = []
        bin_count = 1
        event_in = torch.zeros((2, self.yMax + 1, self.xMax + 1))

        while sIndex < len(self.eventDataFrame):
            while bin_count* frameTiming < self.eventDataFrame["timestamp"].loc[sIndex] and sIndex < len(self.eventDataFrame):
                polarity = self.eventDataFrame["polarity"].loc[sIndex]
                yPos = self.eventDataFrame["y"].loc[sIndex]
                xPos = self.eventDataFrame["x"].loc[sIndex]
                event_in[polarity][self.yMax - yPos][self.xMax - xPos] += 1
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

            pbar.update(1)
        pbar.close()

        self.sbins = binned_event_frames
        return self.sbins
    
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
        return len(self.sbins)

    def __getitem__(self, index) -> any:
        return self.sbins[index]


def f_weight(x):
    return torch.clamp(x, 0, 0.5)

if __name__=="__main__":
    #net = EventVPR(n_in=1,n_out=2)
    eventdt = EventDataset(filepath="./reduce_jAER.txt")
    eventIn = eventdt.buildSpikeBins(bins=500)
    print(f"Max Y={eventdt.yMax}, Max X={eventdt.xMax}")
    
    n_in = (eventdt.xMax + 1) * (eventdt.yMax + 1)


    train_dataloader = DataLoader(eventdt, batch_size=1, shuffle=True)


    net = EventVPR(n_in=n_in,n_out=2, w_mean=0.2, epochs=2000, training_data=train_dataloader)
    net.train()

    learner = learning.STDPLearner(step_mode='s', synapse=net.net[1], sn=net.net[2], tau_post=2., tau_pre=2., f_post=f_weight, f_pre=f_weight)
    stdp_optimizer = torch.optim.SGD(net.net.parameters(), lr=1e-3, momentum=0.)

    net.stdp_training(learner=learner, optimizer=stdp_optimizer)
