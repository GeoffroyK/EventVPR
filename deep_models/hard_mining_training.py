import torch 
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import random
from spikingjelly.clock_driven import neuron, surrogate, functional
from deep_models.training_datasets import TripletVPRDataset, VizDADataset, DATripletVPRDataset, VizTripletDataset, VizDATripletDataset, HardTripletDataset, HardDATripletDataset
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, auc
from deep_models.triplet_mining import HardTripletLoss
from utils.visualisation import plot_recalln

cropping = (0, 0)

# Seeding for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if using multi-GPU
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SNNEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(236),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        # 176128 full resolution
        # 131328 cropping 20, 20
        # 155648 cropping 0, 20
        # 135168 cropping 0, 40 
        self.fc = nn.Linear(86016, output_dim)
        self.sn_final = neuron.IFNode(surrogate_function=surrogate.ATan(), v_threshold=float("inf"), v_reset=0.)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sn_final(x)
        x = self.sn_final.v.detach()
        return x

def to_dataloader(dataset, batch_size=8, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def training_step(model, criterion, optimizer, anchor, target):
    model.train()
    optimizer.zero_grad()
    model(anchor)
    anchor_output = model.sn_final.v
    functional.reset_net(model)
    training_loss, training_acc = criterion(target, anchor_output)
    training_loss.backward()
    optimizer.step()

    return training_loss.item(), training_acc.item()

def validation_step(model, criterion, anchor, target):
    model.eval()
    with torch.no_grad():
        model(anchor)
        anchor_output = model.sn_final.v
        functional.reset_net(model)
        val_loss, val_acc = criterion(target, anchor_output)

    return val_loss.item(), val_acc.item()

def train_network(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=10):
    model.to(device)
    embedding_frames = []  # Store embeddings for animation
    labels = []

    print(f"Using device: {device}")
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_val_loss = 0
        epoch_val_acc = 0
        train_steps = 0
        val_steps = 0

        # Training loop
        for anchor, target in tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}'):
            anchor, target = anchor.to(device), target.to(device)
            loss, acc = training_step(model, criterion, optimizer, anchor, target)
            epoch_train_loss += loss
            epoch_train_acc += acc
            train_steps += 1

        # Validation loop
        for anchor, target in tqdm(test_dataloader, desc=f'Validation Epoch {epoch+1}'):
            anchor, target = anchor.to(device), target.to(device)
            loss, acc = validation_step(model, criterion, anchor, target)
            epoch_val_loss += loss
            epoch_val_acc += acc
            val_steps += 1

        # Calculate epoch metrics
        avg_train_loss = epoch_train_loss / train_steps 
        avg_train_acc = epoch_train_acc / train_steps
        avg_val_loss = epoch_val_loss / val_steps
        avg_val_acc = epoch_val_acc / val_steps

        print(f"Epoch {epoch+1}")
        print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}")
        print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")

    return model, embedding_frames, labels

def visualize_embeddings(model, viz_dataloader, device, n_samples=1000, perplexity=2, title="t-SNE Visualization of Brisbane places embeddings", n_places=10):
    """
    Create t-SNE visualization of the network embeddings using the validation dataset
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for anchor, label in viz_dataloader:
            anchor = anchor.float().to(device)
            # Forward pass
            model(anchor)
            batch_embedding = model.sn_final.v.cpu().numpy()
            embeddings.append(batch_embedding)
            functional.reset_net(model)
            
            labels.extend(label)

            if len(labels) >= n_samples:
                break

    # Concatenate all embeddings and convert labels to numpy array
    embeddings = np.concatenate(embeddings, axis=0)[:n_samples]
    labels = np.array(labels[:n_samples])

    palette = np.array(sns.color_palette("hsv", n_places))
    # Map labels to palette
    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_to_index[label] for label in labels])

    # Apply t-SNE of the real embeddings
   
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=palette[y_numeric], alpha=0.6, s=20)
    
    # Create the plot
    # plt.figure(figsize=(10, 10))
    # scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
    #                      c=labels, cmap='tab10', alpha=0.6, s=20)
    
    plt.colorbar(scatter, label='Place ID')
    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
    plt.close()
    return embeddings_2d, labels

def crop_input(input: torch.Tensor, cropping: tuple):
    '''
    Crop the input tensor to the specified cropping dimensions
    '''
    b, c, h, w = input.shape
    ch, cw = cropping
    return input[:, :, ch:h-ch, cw:w-cw]

def collect_embeddings(model, dataset, device):
    embeddings = []
    labels = []
    with torch.no_grad():
        for anchor, label in dataset:
            anchor = anchor.to(device)
            anchor = anchor.float()
            model(anchor)
            embedding = model.sn_final.v
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(label)
            functional.reset_net(model)
            
    embeddings = np.array(embeddings)
    return torch.tensor(embeddings), torch.tensor(labels)

def compute_distance_matrix(reference, prediction, squared=False):
    """
    Compute the 2D matric of distances between all embeddings.
    
    Args:
        embeddings: tensor with shape [batch_size, embedding_dim]
        squared: boolean, whether to return the squared euclidean distance
    Returns:
        distance_matrix: tensor with shape [batch_size, batch_size]
    """
    # Get the dot product between all embeddings
    dot_product = torch.matmul(reference, prediction.T)
    # Get the square norm of each embedding
    square_norm = torch.diag(dot_product)
    # Compute the pair wise distance matrix
    distances = square_norm.unsqueeze(0) - 2 * dot_product + square_norm.unsqueeze(1)
    # Ensure the distance is non-negative
    distances = torch.max(distances, torch.tensor(0.0))
    # If not squared, take the square root
    if not squared:
        mask = (distances == 0).int()
        distances = distances + (mask * 1e-16) # To prevent NaN gradient in 0
        distances = torch.sqrt(distances)
    return distances

def get_distance_matrix(reference, prediction):
    n_points = reference.shape[0]
    distances = torch.zeros(n_points, n_points)
    for ref in range(n_points):
        for pred in range(n_points):
            distances[ref, pred] = torch.sum((reference[ref] - prediction[pred]) ** 2)
    return distances

def calculate_precision_recall(distances):
    # Convert distancs to similariry score
    similarity_matrix = 1 - distances / torch.max(distances)
    
    # Ground truth should be the elements in the diagonal
    y_true = np.eye(similarity_matrix.shape[0]).flatten()
    y_scores = similarity_matrix.flatten()

    # Compute precision_recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    # Compute AUC for PR curve
    pr_auc = auc(recall, precision)
    
    return recall, precision, pr_auc

def calculate_recall_n(distances):
    n_points = distances.shape[0]
    n_values = np.arange(1, n_points+1)    
    recall_at_n = []

    # For each query (row in distances matrix)
    for n in n_values:
        correct_matches = 0
        for query_idx in range(n_points):
            # Get indices sorted by distance (ascending)
            sorted_indices = torch.argsort(distances[query_idx])
            # Get top-n predictions
            top_n_predictions = sorted_indices[:n]
            # Check if the correct match (same index) is in top-n predictions
            if query_idx in top_n_predictions:
                correct_matches += 1
        recall_at_n.append(correct_matches / n_points)
    return recall_at_n

def compute_recall_n(model, reference, predictions, n_bins, device, state=None, ax=None, save=False):
    reference_dataset = TripletVPRDataset(traverses=reference, n_places=10, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d')
    reference_dataset = VizTripletDataset(reference_dataset)
    reference_dataset = to_dataloader(reference_dataset, batch_size=1, shuffle=False)
    reference_embeddings, reference_labels = collect_embeddings(model, reference_dataset, device)
    reference_embeddings = torch.tensor(reference_embeddings)
   
    predictions_embeddings = []
    for prediction in predictions:
        prediction_dataset = TripletVPRDataset(traverses=[prediction], n_places=10, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d')
        prediction_dataset = VizTripletDataset(prediction_dataset)
        prediction_dataset = to_dataloader(prediction_dataset, batch_size=1, shuffle=False)

        current_embeddings, current_labels = collect_embeddings(model, prediction_dataset, device)
        predictions_embeddings.append(current_embeddings)

    for idx, curr_pred_embeddings in enumerate(predictions_embeddings):
        if ax is None:
            fig, (ax_matrices, ax_recall) = plt.subplots(1, 2, figsize=(20, 10))
        else:
            ax_matrices = ax[0]
            ax_recall = ax[1]
        reference_embeddings = reference_embeddings.squeeze(1)
        curr_pred_embeddings = curr_pred_embeddings.squeeze(1)
        ax_matrices, ax_recall = plot_recalln(reference_embeddings, curr_pred_embeddings, title=f"Recall@N for {reference}_{predictions[idx]} + {state}", legend=f"{predictions[idx]}{state}", save=save, ax_matrices=ax_matrices, ax_recall=ax_recall)
    return ax_recall, ax_matrices

if __name__ == "__main__":
    '''
    Hyperparameters
    '''
    n_bins = 24 # Keep bins pair for now
    n_places = 25
    batch_size = 32
    epochs = 60
    n_augmentations = 7
    output_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNNEncoder(input_dim=2 * (n_bins + 1), hidden_dim=128, output_dim=output_dim).to(device) 
    criterion = HardTripletLoss(margin=1.0, squared=True) # Hard triplet loss with margin
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01) # Optimizer with weight decay

    train_traverses = ["sunset1", "sunrise", "daytime", "morning"]
    train_dataset_original = DATripletVPRDataset(traverses=train_traverses, n_places=n_places, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d', augmentations_per_sample=n_augmentations) #7 pour les deux
    #train_dataset_original = TripletVPRDataset(traverses=train_traverses, n_places=10, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d')
    train_dataset = HardDATripletDataset(train_dataset_original)
    train_loader = to_dataloader(train_dataset, batch_size=batch_size)

    test_traverses = ["sunset2", "morning", "sunset1", "sunrise"] 
    #test_dataset_original = DATripletVPRDataset(traverses=test_traverses, n_places=10, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d', augmentations_per_sample=1)
    test_dataset_original = TripletVPRDataset(traverses=test_traverses, n_places=n_places, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d')
    test_dataset = HardTripletDataset(test_dataset_original)
    test_loader = to_dataloader(test_dataset, batch_size=batch_size)

    viz_loader_val = VizTripletDataset(test_dataset_original)
    viz_loader_val = to_dataloader(viz_loader_val, batch_size=1)

    viz_loader_train = VizDATripletDataset(train_dataset_original)
    viz_loader_train = to_dataloader(viz_loader_train, batch_size=1)

    # For computing recall@N before training
    reference_traverse = ["sunset1"]
    predictions_traverse = ["sunset2"]
    

    # Recall@N stuff
    reference_dataset = TripletVPRDataset(traverses=reference_traverse, n_places=n_places, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d')
    reference_dataset = VizTripletDataset(reference_dataset)
    reference_dataset = to_dataloader(reference_dataset, batch_size=1, shuffle=False)

    predictions_dataset = TripletVPRDataset(traverses=predictions_traverse, n_places=n_places, time_window=0.3, n_hist=n_bins, format='pickle', mode='2d')
    predictions_dataset = VizTripletDataset(predictions_dataset)
    predictions_dataset = to_dataloader(predictions_dataset, batch_size=1, shuffle=False)

    # Get histograms
    def get_histograms(dataset):
        histograms = []
        for anchor, _ in dataset:
            histograms.append(anchor)
        # Stack all tensors along dimension 0
        return torch.stack(histograms, dim=0)

    reference_embeddings, _ = collect_embeddings(model, reference_dataset, device)
    reference_embeddings = torch.tensor(reference_embeddings).squeeze(1)

    reference_histograms = get_histograms(reference_dataset)
    predictions_histograms = get_histograms(predictions_dataset)
    distances_histograms = get_distance_matrix(reference_histograms, predictions_histograms)
    # distances_histograms = compute_distance_matrix(reference_histograms, predictions_histograms)
    recall_at_n_histograms = calculate_recall_n(distances_histograms)
    
    
    predictions_embeddings, _ = collect_embeddings(model, predictions_dataset, device)
    predictions_embeddings = torch.tensor(predictions_embeddings).squeeze(1)
    


    # distances_before = compute_distance_matrix(reference_embeddings, predictions_embeddings)
    distances_before = get_distance_matrix(reference_embeddings, predictions_embeddings)
    recall_at_n_before = calculate_recall_n(distances_before)
    
    #ax_recall, ax_matrices = compute_recall_n(model, reference_traverse, predictions_traverse, n_bins, device, state="before", save=False)
    visualize_embeddings(model, viz_loader_val, device, n_samples=len(viz_loader_val), perplexity=10, title="t-SNE Visualisation, Validation Set; before training", n_places=n_places)

    train_network(model, train_loader, test_loader, optimizer, criterion, device, epochs=60)
  
    #compute_recall_n(model, reference_traverse, predictions_traverse, n_bins, device, state="after",  ax=[ax_recall, ax_matrices], save=True)
    visualize_embeddings(model, viz_loader_val, device, n_samples=len(viz_loader_val), perplexity=10, title="t-SNE Visualisation, Validation Set", n_places=n_places)
    visualize_embeddings(model, viz_loader_train, device, n_samples=len(viz_loader_train), perplexity=10, title="t-SNE Visualisation, Training Set", n_places=n_places)

    # Recall@N after training
    reference_embeddings, _ = collect_embeddings(model, reference_dataset, device)
    reference_embeddings = torch.tensor(reference_embeddings).squeeze(1)

    predictions_embeddings, _ = collect_embeddings(model, predictions_dataset, device)
    predictions_embeddings = torch.tensor(predictions_embeddings).squeeze(1)
    
    #distances_after = compute_distance_matrix(reference_embeddings, predictions_embeddings)
    distances_after = get_distance_matrix(reference_embeddings, predictions_embeddings)
    recall_at_n_after = calculate_recall_n(distances_after)

    # Plotting
    fig, (mat_hist, mat_before, mat_after) = plt.subplots(1, 3, figsize=(15, 6))
    
    # Get the global min and max values across all matrices
    vmin = min(distances_histograms.min(), distances_before.min(), distances_after.min())
    vmax = max(distances_histograms.max(), distances_before.max(), distances_after.max())
    
    # Create heatmaps with consistent color scaling
    sns.heatmap(distances_histograms, ax=mat_hist, cmap="Blues", annot=True, vmin=vmin, vmax=vmax, cbar=False)
    sns.heatmap(distances_before, ax=mat_before, cmap="Blues", annot=True, vmin=vmin, vmax=vmax, cbar=False)
    cbar = sns.heatmap(distances_after, ax=mat_after, cmap="Blues", annot=True, vmin=vmin, vmax=vmax,
                       cbar_kws={'label': 'Distance'})
    
    mat_hist.set_title(f"Histograms")
    mat_before.set_title(f"Before Training")
    mat_after.set_title(f"After Training")
    plt.suptitle(f"Distances Matrices q:{reference_traverse[0]} / p:{predictions_traverse[0]}")
    plt.savefig(f"distances_matrices_q{reference_traverse[0]}_p{predictions_traverse[0]}.png")
    plt.tight_layout()
    plt.close()
    
    # Plotting Recall@N
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    n_points = np.arange(1, n_places+1)
    ax.plot(n_points, recall_at_n_before, label="Before training", marker='o')
    ax.plot(n_points, recall_at_n_after, label="After training", marker='o')
    ax.plot(n_points, recall_at_n_histograms, label="Histograms", marker='o')
    from utils.visualisation import plot_chance_recall
    plot_chance_recall(ax, n_places)
    ax.set_title(f"Recall@N for q:{reference_traverse[0]} / p:{predictions_traverse[0]}", fontsize=14, fontweight='bold')
    ax.set_xlabel("N - Number of top correlated candidates")
    ax.set_ylabel("Average Recall@N (%)")
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"recall_at_n_q{reference_traverse[0]}_p{predictions_traverse[0]}.png")
    plt.close()

    # Plotting Precision - Recall
    recall, precision, pr_auc = calculate_precision_recall(distances_after)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve from Distance Matrix')
    plt.legend()
    plt.grid()
    plt.savefig(f"precision_recall_q{reference_traverse[0]}_p{predictions_traverse[0]}.png")