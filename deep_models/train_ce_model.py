import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import os
import vpr_encoder
import torch.nn.init as init
from triplet_cosine_loss import CosineTripletLoss
from training_datasets import VPRDataset, DAVPRDataset
from spikingjelly.clock_driven import neuron, functional
from utils.data_augmentation import event_drop
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


def initialize_weights(model):
    """
    Initialize the weights of the model.

    Args:
        model (nn.Module): The neural network model.
    """
    for layer in model.children():
        if isinstance(layer, nn.Conv3d):
            init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

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

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        functional.reset_net(model)

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

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
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            functional.reset_net(model)

    val_loss = total_loss / len(dataloader)
    val_accuracy = correct_predictions / total_samples

    return val_loss, val_accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_save_path, model_name, scheduler=None):
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

    last_acc = 0
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

        if last_acc < train_accuracy:
            last_acc = train_accuracy
            current_save_path = model_save_path.split('.')[0]
            current_save_path += f"_{epoch}_{train_accuracy:.4f}.pth"
            os.makedirs(os.path.dirname(current_save_path), exist_ok=True)
            torch.save(model.state_dict(), current_save_path)

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
    model = vpr_encoder.EventVPREncoder(in_channels=2, out_channels=32, kernel_size=7, num_places=args.n_places)
    initialize_weights(model)
    # Move model to specified device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    criterion = nn.CrossEntropyLoss()

    # Create dataset and dataloader
    traverses = ["sunset2", "sunrise", "morning", "sunset1"]

    print(f"Building datasets...")

    if args.data_augmentation:
        train_dataset = DAVPRDataset(traverses, n_places=args.n_places, time_window=args.time_window, n_hist=args.n_hist, format=args.data_format)
        test_dataset = DAVPRDataset(["daytime"], n_places=args.n_places, time_window=args.time_window, n_hist=args.n_hist, format=args.data_format)
    else:
        train_dataset = VPRDataset(traverses, n_places=args.n_places, time_window=args.time_window, n_hist=args.n_hist, format=args.data_format)
        test_dataset = VPRDataset(["daytime"], n_places=args.n_places, time_window=args.time_window, n_hist=args.n_hist, format=args.data_format)
        
    print(f"Test dataset created with {len(test_dataset)} samples")
    print(f"Training dataset created with {len(train_dataset)} samples")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Create the directory for saving the model
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    # Train the model using the VPR Dataset
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        model_save_path=args.model_save_path,
        model_name=args.model_name,
        scheduler=scheduler
    )

    # Save the trained model
    torch.save(model.state_dict(), f"{args.model_save_path}_{args.epochs}_final.pth")
    print(f"Model saved to {args.model_save_path}")