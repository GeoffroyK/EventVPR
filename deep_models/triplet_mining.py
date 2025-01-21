import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class HardTripletLoss(nn.Module):
    def __init__(self, margin=1.0, squared=False):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, labels, embeddings):
        return self.batch_hard_triplet_loss(labels, embeddings, self.margin, self.squared)

    def compute_distance_matrix(self, embeddings, squared=False):
        """
        Compute the 2D matric of distances between all embeddings.
        
        Args:
            embeddings: tensor with shape [batch_size, embedding_dim]
            squared: boolean, whether to return the squared euclidean distance
        Returns:
            distance_matrix: tensor with shape [batch_size, batch_size]
        """
        # Get the dot product between all embeddings
        dot_product = torch.matmul(embeddings, embeddings.T)
        # Get the square norm of each embedding
        square_norm = torch.diag(dot_product)
        # Compute the pair wise distance matrix
        distances = square_norm.unsqueeze(0) - 2 * dot_product + square_norm.unsqueeze(1)
        # Ensure the distance is non-negative
        distances = torch.max(distances, torch.tensor(0.0))
        # If not squared, take the square root
        if not squared:
            distances = torch.sqrt(distances)
        return distances

    def get_anchor_positive_triplet_mask(self, labels):
        """
        Return a 2D mask where mask[a, p] is True if a and p are distinct and have the same label.
        """
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        return labels_equal.unsqueeze(2)

    def get_anchor_negative_triplet_mask(self, labels):
        """
        Return a 2D mask where mask[a, n] is True if a and n are distinct and have different labels.
        """
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        return ~labels_equal.unsqueeze(2) # Bitwise NOT

    def batch_hard_triplet_loss(self, labels, embeddings, margin=1.0, squared=False):
        """
        Build the triplet loss over a batch of embeddings.
        """
        # Compute the distance matrix
        pairwise_dist = self.compute_distance_matrix(embeddings, squared=squared)

        # For each anchor, get the hardest positive
        mask_anchor_positive = self.get_anchor_positive_triplet_mask(labels)
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        # For each anchor, get the hardest positive
        hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

        # For each anchor, get the hardest negative
        mask_anchor_negative = self.get_anchor_negative_triplet_mask(labels)
        anchor_negative_dist = mask_anchor_negative * pairwise_dist
        # For each anchor, get the hardest negative
        hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

        # Combine hardest positive and hardest negative
        triplet_loss = hardest_positive_dist - hardest_negative_dist + margin
        triplet_loss = torch.max(triplet_loss, torch.tensor(0.0))
        # Get the mean of the triplet loss
        triplet_loss = torch.mean(triplet_loss)
        return triplet_loss

if __name__ == "__main__":
    labels = torch.tensor([1, 2, 3, 4]) # The labels of the anchors in the batch
    embeddings = torch.tensor([[1, 2, 3], [2, 3], [3, 4], [4, 5]]) # The embedded vectors


    criterion = HardTripletLoss(margin=1.0, squared=False)
    loss = criterion(labels, embeddings)
    print(loss)