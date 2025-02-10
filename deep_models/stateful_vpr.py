import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate
from model import SNNEncoderStateful
from triplet_mining import HardTripletLoss
from training_datasets import DATripletVPRDataset, HardDATripletDataset, TripletVPRDataset, HardTripletDataset, VizTripletDataset
from hard_mining_training import to_dataloader, get_distance_matrix, calculate_recall_n
from tqdm import tqdm
from utils.visualisation import plot_chance_recall
import matplotlib.pyplot as plt
import numpy as np
import random

# Seeding for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if using multi-GPU
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_histograms(dataset):
    histograms = []
    for anchor, _ in dataset:
        histograms.append(anchor)
    # Stack all tensors along dimension 0
    return torch.stack(histograms, dim=0)

def divide_sub_hists(anchor, n_bins):
    # input anchor shape is [B, N_BINS*2 + 1, H, W]
    # output shape is [T, B, 2, H, W] # On and off channels
    B, _, H, W = anchor.shape
    T = n_bins + 1  # Number of time steps
    # Split the input tensor into on and off channels
    # First n_bins+1 channels are ON events, last n_bins channels are OFF events
    on_channels = anchor[:, :T, :, :]  # Shape: [B, T, H, W]
    off_channels = anchor[:, T:-1, :, :]  # Shape: [B, T-1, H, W] 
    # Pad the off channels with zeros to match the time steps
    off_pad = torch.zeros_like(off_channels[:, :1, :, :])  # Create padding of shape [B, 1, H, W]
    off_channels = torch.cat([off_channels, off_pad], dim=1)  # Shape: [B, T, H, W]
    # Stack on and off channels
    result = torch.stack([on_channels, off_channels], dim=2)  # Shape: [B, T, 2, H, W]
    # Permute to get final shape [T, B, 2, H, W]
    result = result.permute(1, 0, 2, 3, 4)
    return result

def collect_embeddings(model, dataset, device):
    embeddings = []
    labels = []
    with torch.no_grad():
        for anchor, label in dataset:
            anchor = divide_sub_hists(anchor, n_bins)
            for step in anchor:
                step = step.to(device)
                step = step.float()
                model(step)
            embedding = model.sn_final.v
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(label)
            functional.reset_net(model)
            functional.reset_net(model)
            
    embeddings = np.array(embeddings)
    return torch.tensor(embeddings), torch.tensor(labels)


def extract_results(reference_traverse, predictions_traverse):
    reference_dataset = TripletVPRDataset(traverses=reference_traverse, n_places=n_places, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d')
    reference_dataset = VizTripletDataset(reference_dataset)
    reference_dataset = to_dataloader(reference_dataset, batch_size=1, shuffle=False)

    predictions_dataset = TripletVPRDataset(traverses=predictions_traverse, n_places=n_places, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d')
    predictions_dataset = VizTripletDataset(predictions_dataset)
    predictions_dataset = to_dataloader(predictions_dataset, batch_size=1, shuffle=False)

    reference_histograms = get_histograms(reference_dataset)
    predictions_histograms = get_histograms(predictions_dataset)
    distances_histograms = get_distance_matrix(reference_histograms, predictions_histograms)
    recall_at_n_histograms = calculate_recall_n(distances_histograms)
        
    reference_embeddings, _ = collect_embeddings(model, reference_dataset, device)
    reference_embeddings = reference_embeddings.clone().detach().squeeze(1)

    predictions_embeddings, _ = collect_embeddings(model, predictions_dataset, device)
    predictions_embeddings = predictions_embeddings.clone().detach().squeeze(1)

    distances_after = get_distance_matrix(reference_embeddings, predictions_embeddings)
    recall_at_n_after = calculate_recall_n(distances_after)

    # Plotting Recall@N
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    n_points = np.arange(1, n_places+1)
    ax.plot(n_points, recall_at_n_histograms, label="Histograms", marker='o')
    ax.plot(n_points, recall_at_n_after, label="After training", marker='o')
    plot_chance_recall(ax, n_places)
    ax.set_title(f"Recall@N for q:{reference_traverse[0]} / p:{predictions_traverse[0]}", fontsize=14, fontweight='bold')
    ax.set_xlabel("N - Number of top correlated candidates")
    ax.set_ylabel("Average Recall@N (%)")
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"STATEFUL_recall_at_n_q{reference_traverse[0]}_p{predictions_traverse[0]}.png")
    plt.close()

def training_step(model, criterion, optimizer, timestep_anchor, target):
    model.train()
    optimizer.zero_grad()
    for anchor in timestep_anchor:
        anchor = anchor.to(device)
        model(anchor)

    anchor_output = model.sn_final.v
    functional.reset_net(model)
    training_loss, training_acc = criterion(target, anchor_output)
    training_loss.backward()
    optimizer.step()
    return training_loss.item(), training_acc.item()

def train_network(model, train_dataloader, optimizer, criterion, device, epochs=10):
    model.to(device)
    embedding_frames = []  # Store embeddings for animation
    labels = []
    tau_values = []
    print(f"Using device: {device}")
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_train_acc = 0
        train_steps = 0

        # Training loop
        for anchor, target in tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}'):
            anchor, target = anchor.to(device), target.to(device)
            
            # Convert the anchor in sub histograms for statefulness
            timestep_anchor = divide_sub_hists(anchor, n_bins)
            
            
            loss, acc = training_step(model, criterion, optimizer, timestep_anchor, target)
            epoch_train_loss += loss
            epoch_train_acc += acc
            train_steps += 1

        # Calculate epoch metrics
        avg_train_loss = epoch_train_loss / train_steps 
        avg_train_acc = epoch_train_acc / train_steps
        
        print(f"Epoch {epoch+1}")
        print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}")

        # Get tau values
        tau_values.append(model.get_parametric_lif_tau())
    #Plot values overtime
    plt.figure(figsize=(12, 6))
    tau_array = np.array(tau_values)
    for i in range(tau_array.shape[1]):  # Plot each neuron's tau value
        plt.plot(range(1, len(tau_values) + 1), tau_array[:, i], 
                label=f'Neuron {i+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Tau Value')
    plt.title('Evolution of Tau Values Over Training')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('tau_values_evolution.png')
    plt.close()
    return model, embedding_frames, labels

'''
Hyperparameters
'''
# Model hyperparameters
epochs = 16
batch_size = 32
learning_rate = 1e-3
output_dim = 128

# VPR hyperparameters
n_places = 10
time_window = 0.3
n_bins = 24
n_augmentations = 1

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNNEncoderStateful(input_dim= 2, output_dim=output_dim).to(device)
criterion = HardTripletLoss(margin=1.0, squared=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Datasets initialization
train_traverses = ["sunset1", "sunrise", "daytime", "morning"]
train_dataset = DATripletVPRDataset(traverses=train_traverses, n_places=n_places, time_window=time_window, n_hist=n_bins, format='pickle', mode='2d', augmentations_per_sample=n_augmentations)
train_dataset = HardDATripletDataset(train_dataset)
train_loader = to_dataloader(train_dataset, batch_size=batch_size)

train_network(model, train_loader, optimizer, criterion, device, epochs=epochs)

# Compute recall@N on the test traverse
reference_traverse = ["sunset1"]
predictions_traverse = ["sunset2"]
extract_results(reference_traverse, predictions_traverse)