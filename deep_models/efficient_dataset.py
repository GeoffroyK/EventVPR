import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import time
from constant_paths import hot_pixels_locations
import tonic
from tqdm import tqdm
from utils.data_augmentation import event_drop

class TripletVPRDataset(Dataset):
    def __init__(self, traverses: list, n_places:int, time_window: float, n_event_bins: int, event_folder_path: str, sensor_size: tuple, mode="2d"):
        self.traverses = traverses
        self.n_places = n_places
        self.time_window = time_window
        self.n_event_bins = n_event_bins
        self.event_folder_path = event_folder_path
        self.sensor_size = sensor_size
        self.hot_pixels_locations = {}
        self.data = []
        self.labels = []
        self.traverses_label = []
        self.mode = mode

        assert self.mode in ["2d", "3d"], "Mode must be either '2d' or '3d'"

        self.hot_pixels_locations = self.get_hot_pixels()
        # Get the event sequence of a place on each traverse
        for traverse_idx, traverse in enumerate(traverses):
            start_time = time.time()
            print(f"Processing traverse {traverse}")
            with tqdm(total=n_places, desc=f"Processing {traverse}") as pbar:   
                for place_num in range(n_places):
                    # Get the event sequence of the place
                    event_seq = self.get_event_sequence_from_file(traverse, place_num)
                    event_seq = self.filter_hot_pixels(event_seq, traverse)
                    self.data.append(self.to_frame_tensor(event_seq))
                    self.labels.append(place_num)
                    self.traverses_label.append(traverse_idx)
                    pbar.update(1)

            elapsed_time = time.time() - start_time
            print(f"Traverse {traverse} processed in {elapsed_time:.2f} seconds")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
    def argmedian(self, x):
        median_value = np.median(x)
        median_index = np.argmin(np.abs(x - median_value))
        return median_index

    def to_frame(self, x):
        frame = torch.zeros(2, self.sensor_size[0], self.sensor_size[1])
        for event in x:
            x = int(event[1])
            y = int(event[2])
            t = float(event[0])
            p = int(event[3])
            frame[p, x, y] += 1
        return frame

    def to_frame_tensor(self, x):
        sub_event_streams = np.array_split(x, self.n_event_bins)
        frame_tensor = torch.zeros(self.n_event_bins, 2, self.sensor_size[0], self.sensor_size[1])
        for i, sub_event_stream in enumerate(sub_event_streams):
            frame_tensor[i] = self.to_frame(sub_event_stream)

        if self.mode == "3d":
            return frame_tensor
        else:
            return frame_tensor.view(self.n_event_bins * 2, self.sensor_size[0], self.sensor_size[1])

    def get_time_window(self, x, tw):
        xt = x[:,0]
        center = self.argmedian(xt)
        center_value = xt[center]
        beginning  = center_value - tw / 2
        end = center_value + tw / 2
        beginning_index = np.argmin(np.abs(xt - beginning))
        end_index = np.argmin(np.abs(xt - end))
        return x[beginning_index:end_index]

    def get_event_sequence_from_file(self, traverse, place_num):
        event_sequence = np.load(f"{self.event_folder_path}/{traverse}_{place_num}_1.0.npy", allow_pickle=True)
        # Keep only the events within the selected time window
        event_sequence = self.get_time_window(event_sequence, self.time_window)
        return event_sequence
    
    def get_hot_pixels(self, hot_pixels=hot_pixels_locations):
        '''
        Filter out hot pixels from the input tensor.
        Hot pixel are detected though this module ()
        and are caused by the 
        '''
        hot_pixels_locations = {}
        for traverse in self.traverses:
            filepath = hot_pixels[traverse]
            hot_pixels_coords = np.loadtxt(filepath, delimiter=',', dtype=int)
            hot_pixels_locations[traverse] = hot_pixels_coords
        return hot_pixels_locations
    
    def filter_hot_pixels(self, events, traverse):
        hot_pixels_coords = self.hot_pixels_locations[traverse]
        keep_mask = ~np.any(np.all(events[:, [1,2]] == hot_pixels_coords[:, None], axis=2), axis=0)
        events = events[keep_mask]
        return events
    
class TripletDAVPRDataset(Dataset):
    def __init__(self, traverses: list, n_places:int, time_window: float, n_event_bins: int, event_folder_path: str, num_augmentations: int, sensor_size: tuple, mode="2d"):
        self.traverses = traverses
        self.n_places = n_places
        self.time_window = time_window
        self.n_event_bins = n_event_bins
        self.event_folder_path = event_folder_path
        self.sensor_size = sensor_size
        self.hot_pixels_locations = {}
        self.data = []
        self.labels = []
        self.traverses_label = []
        self.mode = mode
        self.num_augmentations = num_augmentations

        assert self.mode in ["2d", "3d"], "Mode must be either '2d' or '3d'"

        self.hot_pixels_locations = self.get_hot_pixels()
        # Get the event sequence of a place on each traverse
        for traverse_idx, traverse in enumerate(traverses):
            start_time = time.time()
            print(f"Processing traverse {traverse}")
            with tqdm(total=n_places, desc=f"Processing {traverse}") as pbar:   
                for place_num in range(n_places):
                    # Get the event sequence of the place
                    event_seq = self.get_event_sequence_from_file(traverse, place_num)
                    event_seq = self.filter_hot_pixels(event_seq, traverse)
                    # Add augmentations
                    for _ in range(self.num_augmentations):
                        augmentation = event_drop(event_seq, dims=self.sensor_size[::-1], option=np.random.randint(1,4))
                        self.data.append(self.to_frame_tensor(augmentation))
                        self.labels.append(place_num)
                        self.traverses_label.append(traverse_idx)
                    pbar.update(1)

            elapsed_time = time.time() - start_time
            print(f"Traverse {traverse} processed in {elapsed_time:.2f} seconds")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
    def argmedian(self, x):
        median_value = np.median(x)
        median_index = np.argmin(np.abs(x - median_value))
        return median_index

    def to_frame(self, x):
        frame = torch.zeros(2, self.sensor_size[0], self.sensor_size[1])
        for event in x:
            x = int(event[1])
            y = int(event[2])
            t = float(event[0])
            p = int(event[3])
            frame[p, x, y] += 1
        return frame

    def to_frame_tensor(self, x):
        sub_event_streams = np.array_split(x, self.n_event_bins)
        frame_tensor = torch.zeros(self.n_event_bins, 2, self.sensor_size[0], self.sensor_size[1])
        for i, sub_event_stream in enumerate(sub_event_streams):
            frame_tensor[i] = self.to_frame(sub_event_stream)

        if self.mode == "3d":
            return frame_tensor
        else:
            return frame_tensor.view(self.n_event_bins * 2, self.sensor_size[0], self.sensor_size[1])

    def get_time_window(self, x, tw):
        xt = x[:,0]
        center = self.argmedian(xt)
        center_value = xt[center]
        beginning  = center_value - tw / 2
        end = center_value + tw / 2
        beginning_index = np.argmin(np.abs(xt - beginning))
        end_index = np.argmin(np.abs(xt - end))
        return x[beginning_index:end_index]

    def get_event_sequence_from_file(self, traverse, place_num):
        event_sequence = np.load(f"{self.event_folder_path}/{traverse}_{place_num}_1.0.npy", allow_pickle=True)
        # Keep only the events within the selected time window
        event_sequence = self.get_time_window(event_sequence, self.time_window)
        return event_sequence
    
    def get_hot_pixels(self, hot_pixels=hot_pixels_locations):
        '''
        Filter out hot pixels from the input tensor.
        Hot pixel are detected though this module ()
        and are caused by the 
        '''
        hot_pixels_locations = {}
        for traverse in self.traverses:
            filepath = hot_pixels[traverse]
            hot_pixels_coords = np.loadtxt(filepath, delimiter=',', dtype=int)
            hot_pixels_locations[traverse] = hot_pixels_coords
        return hot_pixels_locations
    
    def filter_hot_pixels(self, events, traverse):
        hot_pixels_coords = self.hot_pixels_locations[traverse]
        keep_mask = ~np.any(np.all(events[:, [1,2]] == hot_pixels_coords[:, None], axis=2), axis=0)
        events = events[keep_mask]
        return events


if __name__ == "__main__":
    traverses = ["sunset1"]
    event_folder_path = "/home/geoffroy/Documents/EventVPR/notebooks/extracted_places/"
    n_places = 10
    time_window = 0.3
    n_hist = 10
    sensor_size = (346, 260)
    mode = "2d"
    #dataset = TripletVPRDataset(traverses, n_places, time_window, n_hist, event_folder_path, sensor_size, mode=mode)
    dataset = TripletDAVPRDataset(traverses, n_places, time_window, n_hist, event_folder_path, num_augmentations=5, sensor_size=sensor_size, mode=mode)