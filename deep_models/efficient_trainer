import torch 
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import random
from spikingjelly.clock_driven import functional
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, auc
from triplet_mining import HardTripletLoss
from hard_mining_training import SNNEncoder, get_distance_matrix, calculate_recall_n
from efficient_dataset import TripletVPRDataset, TripletDAVPRDataset

def init_seed(seed=42):
    print(f"Initializing seed: {seed}")
    # Seeding for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_network(net, criterion, optimizer, train_loader, device, epochs):
    net.train()
    train_loss = 0
    train_losses = []
    for epoch in tqdm(range(epochs), desc='Training epochs'):
        for batch_idx, (anchor, target) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            anchor = anchor.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = net(anchor)
            loss, _ = criterion(target, output)
            loss.backward()
            optimizer.step()
            functional.reset_net(net)
            train_loss += loss.item()
        train_losses.append(train_loss / batch_idx)
    return train_loss

if __name__ == "__main__":
    init_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ''' HYPERPARAMETERS '''
    train_traverses = ["sunset1", "morning", "daytime", "sunrise"]
    test_traverse = ["sunset2"]
    n_event_bins = 24
    batch_size = 32
    epochs = 5
    n_places = 10
    time_window = 0.3
    mode = "2d"
    sensor_size = (346, 260)
    lr=10e-3
    output_dim = 128
    # event_folder_path = "/home/geoffroy/Documents/EventVPR/notebooks/extracted_places/"
    event_folder_path = "/home/geoffroy/Documents/EventVPR/output/timewindows/"
    '''================='''

    net = SNNEncoder(input_dim=n_event_bins*2, hidden_dim=128, output_dim=128).to(device)
    criterion = HardTripletLoss(margin=1.0, squared=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    
    train_dataset = TripletDAVPRDataset(
        traverses=train_traverses,
        n_places=n_places,
        time_window=time_window,
        n_event_bins=n_event_bins,
        event_folder_path=event_folder_path,
        num_augmentations=7,
        sensor_size=sensor_size,
        mode=mode
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_network(net, criterion, optimizer, train_loader, device, epochs)

    def get_embeddings(net, dataset, device):
        net.eval()
        embeddings = []
        labels = []
        for batch_idx, (anchor, target) in tqdm(enumerate(dataset), desc=f'Embedding dataset', leave=False):
            anchor = anchor.to(device)
            target = torch.tensor([target])
            target = target.to(device)
            anchor = anchor.unsqueeze(0)
            output = net(anchor)
            embeddings.append(output.detach().cpu().numpy())
            labels.append(target)
            functional.reset_net(net)
        embeddings = np.concatenate(embeddings, axis=0)
        return torch.tensor(embeddings), torch.tensor(labels)
    
    query_dataset = TripletVPRDataset(
        traverses= ["sunset2"],
        n_places=n_places,
        time_window=time_window,
        n_event_bins=n_event_bins,
        event_folder_path=event_folder_path,
        sensor_size=sensor_size,
        mode=mode
    )

    pred_dataset = TripletVPRDataset(
        traverses= ["sunset1"],
        n_places=n_places,
        time_window=time_window,
        n_event_bins=n_event_bins,
        event_folder_path=event_folder_path,
        sensor_size=sensor_size,
        mode=mode
    )

    query_embeddings, query_labels = get_embeddings(net, query_dataset, device)
    pred_embeddings, pred_labels = get_embeddings(net, pred_dataset, device)

    distances = get_distance_matrix(pred_embeddings, query_embeddings)
    recall_at_n = calculate_recall_n(distances)



    # Plotting Recall@N
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    n_points = np.arange(1, n_places+1)
    ax.plot(n_points, recall_at_n, label="SpikeVPR", marker='o')
    from utils.visualisation import plot_chance_recall
    plot_chance_recall(ax, n_places)
    ax.set_title(f"Recall@N for q:sunset2 / p:sunset1", fontsize=14, fontweight='bold')
    ax.set_xlabel("N - Number of top correlated candidates")
    ax.set_ylabel("Average Recall@N (%)")
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()