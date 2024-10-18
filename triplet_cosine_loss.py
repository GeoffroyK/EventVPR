import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineTripletLoss(nn.Module):
    """
    CosineTripletLoss implements the triplet loss using cosine similarity.

    This loss function is designed to learn embeddings by minimizing the distance
    between an anchor and a positive sample, while maximizing the distance between
    the anchor and a negative sample.

    The cosine similarity is used as the distance metric, which measures the cosine
    of the angle between two vectors. This makes the loss invariant to the magnitude
    of the embeddings and only considers their direction.

    Args:
        margin (float): The margin in the triplet loss formula. Default is 0.2.

    Forward arguments:
        anchor (torch.Tensor): The anchor embeddings.
        positive (torch.Tensor): The positive embeddings.
        negative (torch.Tensor): The negative embeddings.
        p (int): The norm degree for input normalization.

    Returns:
        torch.Tensor: The computed triplet loss.
    """
    def __init__(self, margin=0.2):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, p):
        # Normalize the embeddings for cosine similarity
        anchor = F.normalize(anchor, p, dim=1)
        positive = F.normalize(positive, p, dim=1)
        negative = F.normalize(negative, p, dim=1)

        # Compute cosine similarity
        positive_sim = F.cosine_similarity(anchor, positive)
        negative_sim = F.cosine_similarity(anchor, negative)

        # We want to maximise disimilarity with the positive sim, so 1 - sim
        loss = torch.clamp(1 - positive_sim - (1 - negative_sim) + self.margin, min = 0.0)

        # Mean value among the batches
        return loss.mean()
