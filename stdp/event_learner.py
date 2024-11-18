import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from spikingjelly.activation_based import neuron, learning, functional, surrogate, layer


class EventLearner(nn.Module):
    '''
    
    '''
    def __init__(self, num_output):
        super().__init__()

        self.num_output = num_output
        self.layers= nn.Sequential(
            # layer.Conv2d(1, 16, kernel_size=8, padding=1, stride=2),
            # neuron.LIFNode(tau=2.),
            # layer.Conv2d(3, 6, kernel_size=4, padding=1, stride=2),
            # neuron.LIFNode(tau=2.),
            # layer.Flatten(start_dim=0),
            # layer.Linear(16*171*128,num_output,bias=False),
            # neuron.LIFNode(tau=2.)

            layer.Conv2d(1, 8, kernel_size=8, padding=1, stride=2),
            neuron.LIFNode(tau=2., v_threshold=1.),
            layer.Conv2d(8, 8, kernel_size=4, padding=1, stride=2),
            neuron.LIFNode(tau=2., v_threshold=1.),
            layer.Conv2d(8, 8, kernel_size=2, padding=0, stride=1),
            neuron.LIFNode(tau=2., v_threshold=15.),
            layer.Flatten(),
            layer.Linear(84*63 * 8,self.num_output, bias=False),
            neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan()),
        )
    
    def forward(self, x):
        return self.layers(x)


class TimeSurfaceDataset(Dataset):
    def __init__(self, timeSurfaceDict):
        super().__init__()
        self.timeSurfaceDict = timeSurfaceDict
        self.keys = list(self.timeSurfaceDict.keys())
        self.traverse = self.keys[0] #Default traverse

    def __len__(self):
        return len(self.timeSurfaceDict[self.traverse])
    
    def __getitem__(self, idx):    
        image = self.timeSurfaceDict[self.traverse][idx]
        label = idx
        return image, label 

    def set_traverse(self, traverse: str):
        if traverse in self.keys:
            self.traverse = traverse
        else:
            print(f"Error, selected traverse does not exist")

class EventDataset(Dataset):
    '''
    
    '''
    def __init__(self, eventDict):
        super().__init__()
        self.eventDict = eventDict
        self.keys = list(self.eventDict.keys())
        self.traverse = self.keys[0] #Default traverse

    def __len__(self):
        total_length = 0
        for key in self.keys:
            total_length += len(self.eventDict[key])
        return total_length
    
    def __getitem__(self, idx):
        image = self.eventDict[self.traverse][idx]
        label = idx
        return image, label 
    
    def set_traverse(self, traverse: str):
        if traverse in self.keys:
            self.traverse = traverse
        else:
            print(f"Error, selected traverse does not exist")


def surrogate_training(net, start_epoch, epochs, w_mean, training_data, num_classes, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}.")
    for epoch in range(start_epoch, epochs):
        net.train()

        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in training_data:
            optimizer.zero_grad()
            img = img.to(device).float()
            label = label.to(device)

            label_onehot = F.one_hot(label, num_classes).float()

            out_firing_rate = net(img)
            loss = F.mse_loss(out_firing_rate, label_onehot)
            loss.backward()
            optimizer.step()

            functional.reset_net(net)

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()

        # The correct rate is calculated as follows. The subscript i of the neuron with the highest firing rate in the output layer is considered as the result of classification. From spiking jelly documentation
        train_acc += (out_firing_rate.argmax(0) == label).float().sum().item()

        end_time = time.time()
        print(f"Epoch {epoch}/{epochs}, time:{end_time-start_time:4f}, loss:{train_loss:.4f}, accuracy {train_acc:.2f}")
    
    return net


def  stdp_training(net, epochs, w_mean, training_data, optimizer):
    # Weight control for STDP pass
    def f_weight(x):
        return torch.clamp(x, 0, 1)
    
    # Check and define the training device (CUDA GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init tqdm bar for training progress
    pbar = tqdm(total=epochs,
        desc=f"Training of model with STDP on {device}",
        position=0)
    

    stdp_learners = []
    # Initial weights of the network on the Linear layer
    for i, layer in enumerate(net.layers.children()):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight.data, mean=w_mean)    
            learner = learning.STDPLearner(step_mode='s', synapse=layer, sn=net.layers[i+1], tau_post=2., tau_pre=2., f_post=f_weight, f_pre=f_weight)
            stdp_learners.append(learner)
    w = []
    out_spike = []
    w.append(net.layers[-2].weight.detach().numpy())

    for _ in range(epochs):
        net.train()
        winner_index = 0
        for input, label in training_data:

            optimizer.zero_grad()
            input = input.to(device)
            input = input.unsqueeze(0).float()

            out_spike = (net(input).detach())
            winner_index = torch.argmax(out_spike).item()
            for learner in stdp_learners:
                learner.step(on_grad=True)
                optimizer.step()
            
            functional.reset_net(net)

            for learner in stdp_learners:
                learner.reset()
        pbar.update(1)
        w.append(net.layers[-2].weight.detach().numpy())

    pbar.close()
    return net, w

if __name__ == "__main__":
    num_output = 2
    # Neural Net    
    net = EventLearner(num_output=num_output)
    print(net.layers)
    # Dataloader
    eventSurfaces = np.load("time_surfaces_tau0.2.npy", allow_pickle=True).item()
    
    traverses = ["sunset1", "sunset2", "morning", "sunrise", "daytime"]
    for traverse in traverses:
        eventSurfaces[traverse] = eventSurfaces[traverse][:num_output]

    dataset = TimeSurfaceDataset(eventSurfaces)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    stdp_optimizer = torch.optim.SGD(net.layers.parameters(), lr=1e-3, momentum=0.)
    net, w = stdp_training(net, 100, w_mean=0.5, training_data=train_dataloader, optimizer=stdp_optimizer)

    optimizer = torch.optim.Adam(net.layers.parameters(), lr=1e-3)
    #net = surrogate_training(net, 0, 50, 0.5, train_dataloader, num_output, optimizer)

    # plt.subplot(3,1,1)
    # plt.imshow(net.layers[0].weight[0].detach().numpy().reshape(8,8).T, cmap="inferno")
    # plt.subplot(3,1,2)
    # plt.imshow(net.layers[0].weight[1].detach().numpy().reshape(8,8).T, cmap="inferno")
    # plt.subplot(3,1,3)
    # plt.imshow(net.layers[0].weight[2].detach().numpy().reshape(8,8).T, cmap="inferno")
    # plt.tight_layout()
    # plt.show()
    
    # Tracer les caractéristiques de la première couche de convolution
    fig, axes = plt.subplots(2, 4, figsize=(6, 6))
    fig.suptitle("Caractéristiques de la première couche de convolution")

    for i in range(8):
        row = i // 4
        col = i % 4
        feature = net.layers[0].weight[i].detach().numpy().reshape(8, 8)
        axes[row, col].imshow(feature.T, cmap="inferno")
        axes[row, col].set_title(f"Filtre {i+1}")
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


    test_acc = 0
    for traverse in traverses:
        for place in range(num_output):
            net.layers[-1] = neuron.LIFNode(tau=2., v_threshold=math.inf)
            # print(f"{traverse}, place {place} : {net(torch.tensor(eventSurfaces[traverse][place]).unsqueeze(0).float()).detach()}")
            # print(f"{place} == {torch.argmax(net.layers[-1].v).item()}")
            if place == torch.argmax(torch.tensor(net.layers[-1].v)).item():
                test_acc += 1
    print(f"test_acc: {test_acc/(len(traverses) * num_output):2f}")