import torch
import torch.nn as nn
from itertools import product
from deep_models.training_datasets import VPRDataset, VoxelVPRDataset
#from spikingjelly.activation_based import functional
from spikingjelly.clock_driven import functional

def cosine_similarity(a, b):
    return nn.functional.cosine_similarity(a, b, dim=1)

def euclidean_distance(a, b):
    return torch.norm(a - b, dim=1)

def cosine_distance(a, b):
    return 1 - cosine_similarity(a, b)

def euclidean_similarity(a, b):
    return 1 / (1 + euclidean_distance(a, b))


def create_cosine_similarity_matrix(model, train_traverses, test_traverses, n_bins, time_window, n_places, format, event_representation):
    # Load all datasets depending on the event representation
    train_datasets = {}
    test_datasets = {}

    for train_traverse in train_traverses:
        print(f"processing {train_traverse}")
        if event_representation == "histogram":
            train_dataset = VPRDataset([train_traverse], n_hist=n_bins, n_places=n_places, time_window=time_window, format=format, mode='2d')
        elif event_representation == "voxel":
            train_dataset = VoxelVPRDataset([train_traverse], n_places=n_places, time_window=time_window, n_bins=n_bins, format=format, dims=(260,346))
        train_datasets[train_traverse] = train_dataset

    for test_traverse in test_traverses:
        print(f"processing {test_traverse}")
        if event_representation == "histogram":
            test_dataset = VPRDataset([test_traverse], n_hist=n_bins, n_places=n_places, time_window=time_window, format=format, mode='2d')
        elif event_representation == "voxel":
            test_dataset = VoxelVPRDataset([test_traverse], n_places=n_places, time_window=time_window, n_bins=n_bins, format=format, dims=(260,346))
        test_datasets[test_traverse] = test_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the number of combinations of train and test traverses
    n_combinations = len(list(product(train_traverses, test_traverses)))
    similarity_matrix = torch.zeros(n_combinations, n_places, n_places)
        
    for i, (train_traverse, test_traverse) in enumerate(product(train_traverses, test_traverses)):
        print(f"Computing sim matrix between {train_traverse}, {test_traverse}")
        train_dataset = train_datasets[train_traverse]
        test_dataset = test_datasets[test_traverse]
        for x_index, (x_place, label) in enumerate(train_dataset):
            x_place = x_place.to(device)
            x_embedding = model(x_place.unsqueeze(0))
            x_embedding = model.fc[-1].v
            functional.reset_net(model)
            for y_index, (y_place, label) in enumerate(test_dataset):
                y_place = y_place.to(device)
                y_embedding = model(y_place.unsqueeze(0))
                y_embedding = model.fc[-1].v
                functional.reset_net(model)
                similarity_matrix[i, x_index, y_index] = cosine_similarity(x_embedding, y_embedding)
    return similarity_matrix

def create_cosine_same_matrix(model, traverse, n_bins, time_window, n_places, format, event_representation):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if event_representation == "histogram":
        dataset = VPRDataset([traverse], n_hist=n_bins, n_places=n_places, time_window=time_window, format=format, mode='2d')
    elif event_representation == "voxel":
        dataset = VoxelVPRDataset([traverse], n_places=n_places, time_window=time_window, n_bins=n_bins, format=format, dims=(260,346))
    
    similarity_matrix = torch.zeros(n_places, n_places)
    
    for place_idx in range(n_places):
        place_tensor, _ = dataset[place_idx]
        # Add batch dimension and move to device
        place_tensor = place_tensor.unsqueeze(0).to(device)
        place_embedding = model(place_tensor)
        place_embedding = model.fc[-1].v
        functional.reset_net(model)
        for other_place_idx in range(n_places):
            other_place_tensor, _ = dataset[other_place_idx]
            # Add batch dimension and move to device
            other_place_tensor = other_place_tensor.unsqueeze(0).to(device)
            other_place_embedding = model(other_place_tensor)
            other_place_embedding = model.fc[-1].v
            similarity_matrix[place_idx, other_place_idx] = cosine_similarity(place_embedding, other_place_embedding)
            functional.reset_net(model)
    return similarity_matrix.unsqueeze(0)

def create_euclidean_similarity_matrix(model, train_traverses, test_traverses, n_bins, time_window, n_places, format, event_representation):
    # Load all datasets depending on the event representation
    train_datasets = {}
    test_datasets = {}

    for train_traverse in train_traverses:
        print(f"processing {train_traverse}")
        if event_representation == "histogram":
            train_dataset = VPRDataset([train_traverse], n_hist=n_bins, n_places=n_places, time_window=time_window, format=format, mode='2d')
        elif event_representation == "voxel":
            train_dataset = VoxelVPRDataset([train_traverse], n_places=n_places, time_window=time_window, n_bins=n_bins, format=format, dims=(260,346))
        train_datasets[train_traverse] = train_dataset

    for test_traverse in test_traverses:
        print(f"processing {test_traverse}")
        if event_representation == "histogram":
            test_dataset = VPRDataset([test_traverse], n_hist=n_bins, n_places=n_places, time_window=time_window, format=format, mode='2d')
        elif event_representation == "voxel":
            test_dataset = VoxelVPRDataset([test_traverse], n_places=n_places, time_window=time_window, n_bins=n_bins, format=format, dims=(260,346))
        test_datasets[test_traverse] = test_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the number of combinations of train and test traverses
    n_combinations = len(list(product(train_traverses, test_traverses)))
    similarity_matrix = torch.zeros(n_combinations, n_places, n_places)
        
    for i, (train_traverse, test_traverse) in enumerate(product(train_traverses, test_traverses)):
        print(f"Computing sim matrix between {train_traverse}, {test_traverse}")
        train_dataset = train_datasets[train_traverse]
        test_dataset = test_datasets[test_traverse]
        for x_index, (x_place, label) in enumerate(train_dataset):
            x_place = x_place.to(device)
            x_embedding = model(x_place.unsqueeze(0))
            x_embedding = model.fc[-1].v
            functional.reset_net(model)
            for y_index, (y_place, label) in enumerate(test_dataset):
                y_place = y_place.to(device)
                y_embedding = model(y_place.unsqueeze(0))
                y_embedding = model.fc[-1].v
                functional.reset_net(model)
                similarity_matrix[i, x_index, y_index] = euclidean_similarity(x_embedding, y_embedding)
    return similarity_matrix

def create_euclidean_same_matrix(model, traverse, n_bins, time_window, n_places, format, event_representation):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if event_representation == "histogram":
        dataset = VPRDataset([traverse], n_hist=n_bins, n_places=n_places, time_window=time_window, format=format, mode='2d')
    elif event_representation == "voxel":
        dataset = VoxelVPRDataset([traverse], n_places=n_places, time_window=time_window, n_bins=n_bins, format=format, dims=(260,346))
    
    similarity_matrix = torch.zeros(n_places, n_places)
    
    for place_idx in range(n_places):
        place_tensor, _ = dataset[place_idx]
        # Add batch dimension and move to device
        place_tensor = place_tensor.unsqueeze(0).to(device)
        place_embedding = model(place_tensor)
        place_embedding = model.fc[-1].v
        functional.reset_net(model)
        for other_place_idx in range(n_places):
            other_place_tensor, _ = dataset[other_place_idx]
            # Add batch dimension and move to device
            other_place_tensor = other_place_tensor.unsqueeze(0).to(device)
            other_place_embedding = model(other_place_tensor)
            other_place_embedding = model.fc[-1].v
            similarity_matrix[place_idx, other_place_idx] = euclidean_similarity(place_embedding, other_place_embedding)
            functional.reset_net(model)
    return similarity_matrix.unsqueeze(0)