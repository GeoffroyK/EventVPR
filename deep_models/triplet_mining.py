"""
Pytorch implementation of HardTripletLoss initially implemented in Tensorflow in :
https://github.com/omoindrot/tensorflow-triplet-loss/tree/master
"""
import torch
import torch.nn as nn

class HardTripletLoss(nn.Module):
    def __init__(self, margin=1.0, squared=False):
        super(HardTripletLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.margin = margin
        self.squared = squared
        
    def forward(self, labels, embeddings):
        return self.batch_hard_triplet_loss(labels, embeddings, self.margin, self.squared)

    def get_valid_triplet_mask(self, labels):
        """
        Return a 1D mask where mask[i] is True if the anchor i has at least
        one valid positive and one valid negative.
        Shape: [batch_size]
        """
        # Get positive and negative masks
        pos_mask = self.get_anchor_positive_triplet_mask(labels)
        neg_mask = self.get_anchor_negative_triplet_mask(labels)
        
        # Check if each anchor has at least one positive and one negative
        valid_anchors = (torch.sum(pos_mask, dim=1) > 0) & (torch.sum(neg_mask, dim=1) > 0)
        return valid_anchors.to(self.device)

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
            mask = (distances == 0).int()
            distances = distances + (mask * 1e-16) # To prevent NaN gradient in 0
            distances = torch.sqrt(distances)
        return distances.to(self.device)

    def get_anchor_positive_triplet_mask(self, labels):
        """
        Return a 2D mask where mask[a, p] is True if a and p are distinct and have the same label.
        Shape: [batch_size, batch_size]
        """
        # Get a mask for same labels
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        # Get a mask for distinct pairs
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        # Combine both masks
        mask = labels_equal & indices_not_equal
        return mask.to(self.device)

    def get_anchor_negative_triplet_mask(self, labels):
        """
        Return a 2D mask where mask[a, n] is True if a and n are distinct and have different labels.
        Shape: [batch_size, batch_size]
        """
        # Get a mask for different labels
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        # Get a mask for distinct pairs
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        # Combine both masks
        mask = (~labels_equal) & indices_not_equal
        return mask.to(self.device)

    def batch_hard_triplet_loss(self, labels, embeddings, margin=1.0, squared=False):
        """
        Build the triplet loss over a batch of embeddings.
        """        
        # Compute the distance matrix
        pairwise_dist = self.compute_distance_matrix(embeddings, squared=squared)
        valid_triplets = self.get_valid_triplet_mask(labels)
        pairwise_dist = pairwise_dist[valid_triplets]

        identity_mask = torch.eye(labels.size(0))[valid_triplets].bool()

        # For each anchor, get the hardest positive
        mask_anchor_positive = self.get_anchor_positive_triplet_mask(labels)[valid_triplets]

        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        # For each anchor, get the hardest positive 
        hardest_positive_dist, _ = torch.max(anchor_positive_dist - identity_mask.int(), dim=1, keepdim=True)

        # For each anchor, get the hardest negative
        mask_anchor_negative = self.get_anchor_negative_triplet_mask(labels)[valid_triplets]

        # Change False values of mask to very high value
        test_negative_mask = ~mask_anchor_negative
        test_negative_mask = test_negative_mask + identity_mask
        test_negative_mask = test_negative_mask.int() * 1e16


        anchor_negative_dist = test_negative_mask + pairwise_dist #+ (~mask_anchor_positive + identity_mask)
        # For each anchor, get the hardest negative
        hardest_negative_dist, _ = torch.min(anchor_negative_dist + identity_mask, dim=1, keepdim=True)
        
        # Combine hardest positive and hardest negative
        triplet_loss = hardest_positive_dist - hardest_negative_dist + margin
        triplet_loss = torch.max(triplet_loss, torch.tensor(0.0))



        # Get the mean of the valid triplet
        n_valid_triplets = torch.sum(valid_triplets)
        # If no valid triplets, return 0
        if n_valid_triplets == 0:
            return torch.tensor(0.0)

        # Get the mean of the triplet loss
        triplet_loss = torch.mean(triplet_loss)
        accuracy = torch.mean((hardest_positive_dist + margin < hardest_negative_dist).float())

        return triplet_loss, accuracy

if __name__ == "__main__":
    labels = torch.tensor([1, 1, 2])  # The labels of the anchors in the batch
    embeddings = torch.tensor([[1, 1], [1, 1], [1, 2]])  # The embedded vectors
    criterion = HardTripletLoss(margin=1.0, squared=True)
    
    
    loss, acc = criterion(labels, embeddings)
    print("\nFinal loss:", loss.item())