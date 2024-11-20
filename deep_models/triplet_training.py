import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import os
import vpr_encoder
from triplet_cosine_loss import CosineTripletLoss
from training_datasets import VPRDataset, TripletVPRDataset, DATripletVPRDataset
from spikingjelly.clock_driven import neuron, functional
import torch.nn.functional as F
import wandb

# Set GPU Parameter
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

def init_wandb(project_name, config):
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        project_name (str): Name of the W&B project.
        config (dict): Configuration parameters for the experiment.

    Return=2
        wandb.Run: The initialized W&B run object.
    """
    run = wandb.init(project=project_name, config=config)
    return run

def log_metrics(metrics):
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary containing metrics to be logged.
    """
    wandb.log(metrics)

def finish_wandb():
    """
    Finish the current W&B run.
    """
    wandb.finish()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to run the training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)
        loss = criterion(anchor_output, positive_output, negative_output)
            
        distance_pos = float(F.cosine_similarity(positive_output, anchor_output))
        distance_neg = float(F.cosine_similarity(negative_output, anchor_output))
        training_accuracy = max(0, distance_neg - distance_pos)
        
        loss.backward()
        optimizer.step()
        functional.reset_net(model)

        total_loss += loss.item()
        correct_predictions += training_accuracy
        total_samples += 1
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy

def validate(model, dataloader, criterion, device):
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion: The loss function.
        device: The device to run the validation on.

    Returns:
        tuple: (validation loss, validation accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)
            loss = criterion(anchor_output, positive_output, negative_output)
            
            distance_pos = float(F.cosine_similarity(positive_output, anchor_output))
            distance_neg = float(F.cosine_similarity(negative_output, anchor_output))
            val_accuracy = max(0, distance_neg - distance_pos)

            total_loss += loss.item()
            correct_predictions += val_accuracy
            total_samples += 1
            
            functional.reset_net(model)

    val_loss = total_loss / len(dataloader)
    val_accuracy = correct_predictions / total_samples

    return val_loss, val_accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, scheduler=None):
    """
    Train the model for a specified number of epochs.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to run the training on.
        num_epochs (int): Number of epochs to train for.

    Returns:
        nn.Module: The trained model.
    """
    wandb_config = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "architecture": "CNN",
        "dataset": "Brisbane VPR",
        "epochs": num_epochs
    }
    run = init_wandb(model_name, wandb_config)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        if scheduler != None:
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # W&B Log metrics
        log_metrics({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_acc': val_accuracy
        })
        
    finish_wandb()
    return model

def set_random_seed(device, seed):
    # Set the seed for Python's built-in random module to ensure reproducibility
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train VPR Encoder')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model_save_path', type=str, default='vpr_encoder.pth', help='Path to save the trained model')
    parser.add_argument('--n_places', type=int, default=25, help='Number of places to learn')
    parser.add_argument('--data_format', type=str, choices=['pickle', 'txt'], default='txt', help='Format for reading data: pickle or txt')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA GPU device number (default: 0)')
    parser.add_argument('--data_augmentation', action='store_true', default=False, help="Use data augmentation for training")
    parser.add_argument('--n_hist', type=int, default=20, help='Number of histograms fed to the network')
    parser.add_argument('--time_window', type=float, default=0.06, help='Time window for each histogram in seconds')
    parser.add_argument('--model_name', type=str, default='EventVPR', help='Name of the model for wandb logging')
    parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu') 
    print(f"Training on {device}")
    set_random_seed(device, args.seed)

    # Initialize the model, optimizer, and loss function
    model = vpr_encoder.EmbeddedVPREncoder(in_channels=2, out_channels=32, kernel_size=7, num_places=args.n_places, embedding_size=258)
    # Move model to specified device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #criterion = CosineTripletLoss(margin=0.2)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    # Create dataset and dataloader
    traverses = ["sunset2", "sunrise", "morning", "sunset1"]

    print(f"Building datasets...")

    if args.data_augmentation:
        train_dataset = DATripletVPRDataset(traverses, n_places=args.n_places, time_window=args.time_window, n_hist=args.n_hist, format=args.data_format)
        test_dataset = DATripletVPRDataset(["daytime"], n_places=args.n_places, time_window=args.time_window, n_hist=args.n_hist, format=args.data_format)
    else:
        train_dataset = TripletVPRDataset(traverses, n_places=args.n_places, time_window=args.time_window, n_hist=args.n_hist, format=args.data_format)
        test_dataset = TripletVPRDataset(["daytime"], n_places=args.n_places, time_window=args.time_window, n_hist=args.n_hist, format=args.data_format)
            
    print(f"Training dataset created with {len(train_dataset)} samples")
    print(f"Test dataset created with {len(test_dataset)} samples")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Train the model using the VPR Dataset
    #train_vpr_encoder(model, train_loader, optimizer, criterion, device, args.epochs)
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        model_name=args.model_name,
        scheduler=scheduler
    )

    # Save the trained model
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")
