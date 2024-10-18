# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label, center_of_mass
import numpy as np

class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1, dice_weight=1, pos_weight=None, smooth=1, weight=None):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.dice_loss = WeightedDiceLoss(smooth=smooth, weight=weight)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def compute_centroids(self, mask):
        labeled_mask, num_blobs = label(mask.detach().cpu().numpy())
        centroids = center_of_mass(mask.detach().cpu().numpy(), labeled_mask, range(1, num_blobs + 1))
        centroids = [centroid[:2] for centroid in centroids]
        return np.array(centroids)

    def centroid_loss(self, pred_mask, true_mask):
        pred_mask = pred_mask.float()
        true_mask = true_mask.float()
        true_centroids = self.compute_centroids(true_mask)
        pred_centroids = self.compute_centroids(pred_mask)
        mx = max(len(true_centroids), len(pred_centroids))
        if len(true_centroids) == 0 or len(pred_centroids) == 0:
            return torch.tensor(torch.abs(torch.tensor(len(pred_centroids) - len(true_centroids)) / mx), dtype=torch.float32, requires_grad=True)
        count_loss = torch.abs(torch.tensor((len(pred_centroids) - len(true_centroids)) / mx))
        centroid_distances = []
        for pred_c in pred_centroids:
            distances = np.linalg.norm(true_centroids - pred_c, axis=1)
            min_distance = np.min(distances)
            centroid_distances.append(min_distance)
        return torch.tensor(count_loss + (np.mean(centroid_distances) / len(true_centroids)) / 725, dtype=torch.float32, requires_grad=True)

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        cc_loss = self.centroid_loss(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss + cc_loss
