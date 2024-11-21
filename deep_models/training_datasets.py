import random
import torch
import torch.nn as nn

import deep_models.vpr_encoder as vpr_encoder
from utils.data_augmentation import event_drop
from utils import recalltw
from torch.utils.data import Dataset



class VPRDataset(Dataset):
    """
    A PyTorch Dataset for Visual Place Recognition (VPR) using event data.

    This dataset class processes event sequences from multiple traverses, creating
    histogram tensors for specific places. It's designed to work with the VPR encoder
    for training and evaluation purposes.

    Attributes:
        data (list): A list of tensor representations of histogram sequences.
        labels (list): A list of corresponding place labels for each data item.

    Args:
        traverses (list): A list of traverse identifiers to process.
        n_places (int): The number of places in each traverse.
        time_window (float): The time window for each histogram in seconds.
        n_hist (int): The number of histograms in each sequence.
        format (str): The format of the input data files. Can be either 'pickle' or 'txt'.
                    If 'pickle', loads data from .npy files using np.load().
                    If 'txt', loads data from .txt files using np.loadtxt().
        
    The dataset creates tensor representations for places 0 and 20 in each traverse,
    using the specified time window and number of histograms. Each data item is a
    tensor of shape [C, N_HIST, H, W], where C is the number of channels (2 for ON
    and OFF events), N_HIST is the number of histograms, and H and W are the height
    and width of each histogram.
    """
    def __init__(self, traverses, n_places, time_window, n_hist, format):
        self.data = []
        self.labels = []
        self.traverses = traverses
        self.n_places = n_places
        self.time_window = time_window
        self.n_hist = n_hist
        self.format = format

        for traverse in traverses:
            event_seq = recalltw.get_event_seq(traverse, n_places, 1.0, format)
            for place in range(n_places): 
                hist_seq = recalltw.time_windows_around(event_seq[place], time_window, n_hist)
                tensor = vpr_encoder.convert_hist_tensor(1, hist_seq, [260, 346])
                self.data.append(tensor.squeeze(0))  # Remove batch dimension
                self.labels.append(place)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class DAVPRDataset(Dataset):
    """
    A PyTorch Dataset for Visual Place Recognition (VPR) using event data.

    This dataset class processes event sequences from multiple traverses, creating
    histogram tensors for specific places. It's designed to work with the VPR encoder
    for training and evaluation purposes.

    Attributes:
        data (list): A list of tensor representations of histogram sequences.
        labels (list): A list of corresponding place labels for each data item.

    Args:
        traverses (list): A list of traverse identifiers to process.
        n_places (int): The number of places in each traverse.
        time_window (float): The time window for each histogram in seconds.
        n_hist (int): The number of histograms in each sequence.
        format (str): The format of the input data files. Can be either 'pickle' or 'txt'.
                    If 'pickle', loads data from .npy files using np.load().
                    If 'txt', loads data from .txt files using np.loadtxt().
        
    The dataset creates tensor representations for places 0 and 20 in each traverse,
    using the specified time window and number of histograms. Each data item is a
    tensor of shape [C, N_HIST, H, W], where C is the number of channels (2 for ON
    and OFF events), N_HIST is the number of histograms, and H and W are the height
    and width of each histogram.
    """
    def __init__(self, traverses, n_places, time_window, n_hist, format):
        self.data = []
        self.labels = []
        self.traverses = traverses
        self.n_places = n_places
        self.time_window = time_window
        self.n_hist = n_hist
        self.format = format

        for traverse in traverses:
            event_seq = recalltw.get_event_seq(traverse, n_places, 1.0, format)
            for place in range(n_places): 
                events = recalltw.raw_events_around(event_seq[place], time_window, n_hist)
                self.data.append(events)  # Remove batch dimension
                self.labels.append(place)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        # Drop events on a random tensor of the sequence
        drop = random.randint(0,len(data))
        for index, event_seq in enumerate(data):
            if index == drop:
                data[index] = recalltw.event_histogram(event_drop(event_seq, dims=(346,260)))
            else:
                data[index] = recalltw.event_histogram(event_seq)
        out_tensor = vpr_encoder.convert_hist_tensor(1, data, [260, 346])

        return out_tensor.squeeze(0), self.labels[idx]



class TripletVPRDataset(VPRDataset):
    """
    A PyTorch Dataset for Visual Place Recognition (VPR) using event data in a triplet format.

    This dataset class extends the VPRDataset to provide triplets of data points:
    an anchor, a positive sample (same place as anchor), and a negative sample 
    (different place from anchor). This is particularly useful for training models 
    with triplet loss functions.

    Attributes:
        Inherits all attributes from VPRDataset.

    Args:
        traverses (list): A list of traverse identifiers to process.
        n_places (int): The number of places in each traverse.
        time_window (float): The time window for each histogram in seconds.
        n_hist (int): The number of histograms in each sequence.
        format (str): The format of the input data files. Can be either 'pickle' or 'txt'.

    The dataset creates triplets where:
    - The anchor is a randomly selected data point.
    - The positive sample is another data point from the same place as the anchor.
    - The negative sample is a data point from a different place than the anchor.

    Each item returned by __getitem__ is a tuple of (anchor, positive, negative),
    where each element is a tensor of shape [C, N_HIST, H, W].
    """
    def __init__(self, traverses:list, n_places:int, time_window:float, n_hist:int, format:str):
        super().__init__(traverses, n_places, time_window, n_hist, format)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.labels[idx]

        # Positive sample, random sample with the same label (place)
        positive_idx = torch.randint(len(self.data), (1,))
        while self.labels[positive_idx] != anchor_label:
            positive_idx = torch.randint(len(self.data), (1,))
        positive = self.data[positive_idx]

        # Negative sample, random sample the a different label (place)
        negative_idx = torch.randint(len(self.data), (1,))
        while self.labels[negative_idx] == anchor_label:
            negative_idx = torch.randint(len(self.data), (1,))
        negative = self.data[negative_idx]

        return anchor, positive, negative

class DATripletVPRDataset(DAVPRDataset):
    '''
    This dataset class extends the TripletVPRDataset to provide a more advanced way of generating triplets for training models with triplet loss functions. 

    Attributes:
        Inherits all attributes from TripletVPRDataset.

    Args:
        traverses (list): A list of traverse identifiers to process.
        n_places (int): The number of places in each traverse.
        time_window (float): The time window for each histogram in seconds.
        n_hist (int): The number of histograms in each sequence.
        format (str): The format of the input data files. Can be either 'pickle' or 'txt'.

    The dataset creates triplets where:
    - The anchor is a randomly selected data point with some events dropped to simulate real-world scenarios.
    - The positive sample is another data point from the same place as the anchor, also with some events dropped.
    - The negative sample is a data point from a different place than the anchor, with some events dropped.

    Each item returned by __getitem__ is a tuple of (anchor, positive, negative),
    where each element is a tensor of shape [C, N_HIST, H, W].
 
    TODO: Add parameter to get negative samples geography close to the anchor 
    '''
    def __init__(self, traverses:list, n_places:int, time_window:float, n_hist:int, format:str):
        super().__init__(traverses, n_places, time_window, n_hist, format)

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        positive = self.data[idx].copy()
        negative = self.data[idx].copy()
        anchor_label = self.labels[idx]

        # Drop events on a random tensor of the sequence
        drop = random.randint(0,len(data))
        
        for index, event_seq in enumerate(data):
            if index == drop:
                data[index] = recalltw.event_histogram(event_drop(event_seq, dims=(346,260)))
            else:
                data[index] = recalltw.event_histogram(event_seq)

        # == Create Positive Block == 
        # Positive sample, random sample with the same label (place)
        positive_idx = torch.randint(len(self.data), (1,))
        while self.labels[positive_idx] != anchor_label:
            positive_idx = torch.randint(len(self.data), (1,))
        positive_raw = self.data[positive_idx]
        # Convert to list of histograms
        for index, event_seq in enumerate(positive_raw):
            positive[index] = recalltw.event_histogram(event_seq)

        # == Create Negative Block == 
        # Negative sample, random sample the a different label (place)
        negative_idx = torch.randint(len(self.data), (1,))
        while self.labels[negative_idx] == anchor_label:
            negative_idx = torch.randint(len(self.data), (1,))
        negative_raw = self.data[negative_idx]
        # Convert to list of histograms
        for index, event_seq in enumerate(negative_raw):
            negative[index] = recalltw.event_histogram(event_seq)

        anchor = vpr_encoder.convert_hist_tensor(1, data, [260, 346]).squeeze(0)
        positive = vpr_encoder.convert_hist_tensor(1, positive, [260, 346]).squeeze(0)
        negative = vpr_encoder.convert_hist_tensor(1, negative, [260, 346]).squeeze(0)
        return (anchor, positive, negative)
