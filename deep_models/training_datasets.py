import random
import torch
import torch.nn as nn
import numpy as np

import deep_models.vpr_encoder as vpr_encoder
from utils.data_augmentation import event_drop
from utils import recalltw
from torch.utils.data import Dataset
from utils.voxel_representation import convert_voxel_grid
from deep_models.constant_paths import hot_pixels_locations

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
            event_seq = recalltw.get_event_seq(traverse, n_places, 1.0, format)
            
            for idx, place in enumerate(event_seq):
                event_seq[idx] = self.filter_hot_pixels(traverse, place)

            for place in range(n_places): 
                hist_seq = recalltw.time_windows_around(event_seq[place], time_window, n_hist)
                tensor = vpr_encoder.convert_hist_tensor(1, hist_seq, [260, 346], mode=self.mode)
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
            event_seq = recalltw.get_event_seq(traverse, n_places, 1.0, format)
            for place in range(n_places): 
                events = recalltw.raw_events_around(event_seq[place], time_window, n_hist)
          
                if len(events.shape) ==  1: # We need to unroll the events, several bins
                    for idx, event in enumerate(events):
                        events[idx] = self.filter_hot_pixels(traverse, event)
                else:
                    events = self.filter_hot_pixels(traverse, events.squeeze(0))
                    events = np.expand_dims(events, axis=0)
      
                self.data.append(events)
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
        out_tensor = vpr_encoder.convert_hist_tensor(1, data, [260, 346], mode=self.mode)

        return out_tensor.squeeze(0), self.labels[idx]

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
    def __init__(self, traverses:list, n_places:int, time_window:float, n_hist:int, format:str, mode='3d'):
        super().__init__(traverses, n_places, time_window, n_hist, format, mode)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.labels[idx]

        positive_indices = [i for i, label in enumerate(self.labels) 
                            if label == anchor_label and i != idx]
        negative_indices = [i for i, label in enumerate(self.labels) 
                            if label != anchor_label]
        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)

        return anchor, self.data[positive_idx], self.data[negative_idx]

class DATripletVPRDataset(DAVPRDataset):
    def __init__(self, traverses:list, n_places:int, time_window:float, n_hist:int, format:str, 
                 augmentations_per_sample:int = 5, mode='3d'):
        super().__init__(traverses, n_places, time_window, n_hist, format, mode)
        
        self.processed_data = []
        self.processed_labels = []
        
        print("Preprocessing data with augmentations...")
        for idx, events in enumerate(self.data):
            # Store the original version
            hist_seq = []
            for event_seq in events:
                if event_seq.shape[1] == 4:  # Raw events (Nx4)
                    hist_seq.append(recalltw.event_histogram(event_seq))
                else:  # Already a histogram (260x346x2)
                    hist_seq.append(event_seq)
                    
            tensor = vpr_encoder.convert_hist_tensor(1, hist_seq, [260, 346], mode=self.mode).squeeze(0)
            # If tensor is 4D [2, 1, 260, 346] and we only have one bin, squeeze it to [2, 260, 346]
            if len(tensor.shape) == 4 and tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            self.processed_data.append(tensor)
            self.processed_labels.append(self.labels[idx])
            
            # Create augmented versions
            for _ in range(augmentations_per_sample):
                augmented = events.copy()
                drop = random.randint(0, len(augmented)-1)
                
                hist_seq = []
                for index, event_seq in enumerate(augmented):
                    if index == drop:
                        if event_seq.shape[1] == 4:  # Raw events
                            dropped_events = event_drop(event_seq, dims=(346,260))
                            hist_seq.append(recalltw.event_histogram(dropped_events))
                        else:  # Already a histogram
                            hist_seq.append(event_seq)
                    else:
                        if event_seq.shape[1] == 4:  # Raw events
                            hist_seq.append(recalltw.event_histogram(event_seq))
                        else:  # Already a histogram
                            hist_seq.append(event_seq)
                
                tensor = vpr_encoder.convert_hist_tensor(1, hist_seq, [260, 346], mode=self.mode).squeeze(0)
                # If tensor is 4D [2, 1, 260, 346] and we only have one bin, squeeze it to [2, 260, 346]
                if len(tensor.shape) == 4 and tensor.shape[1] == 1:
                    tensor = tensor.squeeze(1)
                self.processed_data.append(tensor)
                self.processed_labels.append(self.labels[idx])
        
        # Clear the original data to save memory
        self.data = None
        print(f"Preprocessing complete. Total samples: {len(self.processed_data)}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        anchor = self.processed_data[idx]
        anchor_label = self.processed_labels[idx]

        # Find indices of all samples with same label
        positive_indices = [i for i, label in enumerate(self.processed_labels) 
                          if label == anchor_label and i != idx]
        # Find indices of all samples with different labels
        negative_indices = [i for i, label in enumerate(self.processed_labels) 
                          if label != anchor_label]

        # Randomly select positive and negative samples
        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)

        return (anchor, 
                self.processed_data[positive_idx],
                self.processed_data[negative_idx])

    def get_anchor_label(self, idx):
        return self.processed_labels[idx]

class TripletVoxelDataset(Dataset):
    '''
    A PyTorch Dataset for Visual Place Recognition (VPR) using voxel grid event data in a triplet format.

    This dataset class creates triplets of voxel grid representations from event data for training
    with triplet loss functions. The voxel grid representation divides events into spatial and temporal bins,
    preserving temporal information in the data.
    Attributes:
        data (list): List of voxel grid representations
        labels (list): List of place labels corresponding to each voxel grid
        traverses (list): List of traverse identifiers being processed
        n_places (int): Number of places in each traverse
        time_window (float): Duration of each temporal bin in seconds
        n_bins (int): Number of temporal bins in the voxel grid
        format (str): Format of input data files ('pickle' or 'txt')
    Args:
        traverses (list): A list of traverse identifiers to process
        n_places (int): The number of places in each traverse
        time_window (float): Duration of each temporal bin in seconds
    
    '''
    def __init__(self, traverses:list, n_places:int, time_window:float, n_bins:int, format:str, dims:tuple, mode='3d'):
        self.mode = mode
        self.data = []
        self.labels = []
        self.traverses = traverses
        self.n_places = n_places
        self.time_window = time_window
        self.n_bins = n_bins
        self.format = format
        self.dims = dims
        for traverse in self.traverses:
            for place in range(self.n_places):
                event_seq = recalltw.get_event_seq(traverse, self.n_places, 1.0, self.format)
                place_seq = recalltw.get_time_window(event_seq[place], self.time_window * self.n_bins) # Each bin has a duration of time_window
                voxel_grid = convert_voxel_grid(place_seq, self.n_bins, self.dims)
                voxel_grid = torch.tensor(voxel_grid, dtype=torch.float32)
                self.data.append(voxel_grid)
                self.labels.append(place)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.labels[idx]

        positive_indices = [i for i, label in enumerate(self.labels) 
                            if label == anchor_label and i != idx]
        negative_indices = [i for i, label in enumerate(self.labels) 
                            if label != anchor_label]
        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)

        return anchor, self.data[positive_idx], self.data[negative_idx]


class VoxelVPRDataset(Dataset):
    def __init__(self, traverses:list, n_places:int, time_window:float, n_bins:int, format:str, dims:tuple, mode='3d'):
        self.mode = mode
        self.data = []
        self.labels = []
        self.traverses = traverses
        self.n_places = n_places
        self.time_window = time_window
        self.n_bins = n_bins
        self.format = format
        self.dims = dims
        for traverse in self.traverses:
            for place in range(self.n_places):
                event_seq = recalltw.get_event_seq(traverse, self.n_places, 1.0, self.format)
                place_seq = recalltw.get_time_window(event_seq[place], self.time_window * self.n_bins) # Each bin has a duration of time_window
                voxel_grid = convert_voxel_grid(place_seq, self.n_bins, self.dims)
                voxel_grid = torch.tensor(voxel_grid, dtype=torch.float32)
                self.data.append(voxel_grid)
                self.labels.append(place)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class VizDADataset(DAVPRDataset):
    def __init__(self, traverses:list, n_places:int, time_window:float, n_hist:int, format:str, 
                 augmentations_per_sample:int = 5, mode='3d'):
        super().__init__(traverses, n_places, time_window, n_hist, format, mode)
        
        self.processed_data = []
        self.processed_labels = []
        
        print("Preprocessing data with augmentations...")
        for idx, events in enumerate(self.data):
            # Store the original version
            hist_seq = []
            for event_seq in events:
                if event_seq.shape[1] == 4:  # Raw events (Nx4)
                    hist_seq.append(recalltw.event_histogram(event_seq))
                else:  # Already a histogram (260x346x2)
                    hist_seq.append(event_seq)
                    
            tensor = vpr_encoder.convert_hist_tensor(1, hist_seq, [260, 346], mode=self.mode).squeeze(0)
            # If tensor is 4D [2, 1, 260, 346] and we only have one bin, squeeze it to [2, 260, 346]
            if len(tensor.shape) == 4 and tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            self.processed_data.append(tensor)
            self.processed_labels.append(self.labels[idx])
            
            # Create augmented versions
            for _ in range(augmentations_per_sample):
                augmented = events.copy()
                drop = random.randint(0, len(augmented)-1)
                
                hist_seq = []
                for index, event_seq in enumerate(augmented):
                    if index == drop:
                        if event_seq.shape[1] == 4:  # Raw events
                            dropped_events = event_drop(event_seq, dims=(346,260))
                            hist_seq.append(recalltw.event_histogram(dropped_events))
                        else:  # Already a histogram
                            hist_seq.append(event_seq)
                    else:
                        if event_seq.shape[1] == 4:  # Raw events
                            hist_seq.append(recalltw.event_histogram(event_seq))
                        else:  # Already a histogram
                            hist_seq.append(event_seq)
                
                tensor = vpr_encoder.convert_hist_tensor(1, hist_seq, [260, 346], mode=self.mode).squeeze(0)
                # If tensor is 4D [2, 1, 260, 346] and we only have one bin, squeeze it to [2, 260, 346]
                if len(tensor.shape) == 4 and tensor.shape[1] == 1:
                    tensor = tensor.squeeze(1)
                self.processed_data.append(tensor)
                self.processed_labels.append(self.labels[idx])
        
        # Clear the original data to save memory
        self.data = None
        print(f"Preprocessing complete. Total samples: {len(self.processed_data)}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        anchor = self.processed_data[idx]
        anchor_label = self.processed_labels[idx]
        return anchor, anchor_label

class VizTripletDataset(Dataset):
    def __init__(self, triplet_dataset):
        self.triplet_dataset = triplet_dataset
    
    def __len__(self):
        return len(self.triplet_dataset)
    
    def __getitem__(self, idx):
        anchor = self.triplet_dataset.data[idx]
        label = self.triplet_dataset.labels[idx]
        return anchor, label

    def get_original_data(self, n_samples):
        anchors = self.triplet_dataset.data[0:n_samples-1]
        labels = self.triplet_dataset.labels[0:n_samples-1]
        return anchors, labels


class VizDATripletDataset(Dataset):
    def __init__(self, triplet_dataset):
        self.triplet_dataset = triplet_dataset
    def __len__(self):
        return len(self.triplet_dataset)
    
    def __getitem__(self, idx):
        anchor = self.triplet_dataset.processed_data[idx]
        label = self.triplet_dataset.processed_labels[idx]
        return anchor, label

    def get_original_data(self, n_samples):
        anchors = self.triplet_dataset.processed_data[0:n_samples-1]
        labels = self.triplet_dataset.processed_labels[0:n_samples-1]
        return anchors, labels

class HardDATripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        anchor = self.dataset.processed_data[idx]
        label = self.dataset.processed_labels[idx]
        return anchor, label
    
class HardTripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        anchor = self.dataset.data[idx]
        label = self.dataset.labels[idx]
        return anchor, label