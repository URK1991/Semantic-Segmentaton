import torch
import torch as th
import torch.nn as nn
import os
import copy
import torch.nn.functional as F
from torch.nn.functional import dropout
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.measure import label
from torch.nn import init
import torch.nn.init as init
import numpy as np
from torch import Tensor
from einops import rearrange, repeat
from math import sqrt
from functools import partial
from torch import nn, einsum
import random
from PIL import ImageFilter
from scipy.ndimage import label, center_of_mass


class CustomData(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            mask_dir (str): Path to the directory with masks.
            image_transform (callable, optional): Transform to be applied to the images.
            mask_transform (callable, optional): Transform to be applied to the masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path)
        image = image.filter(ImageFilter.UnsharpMask(radius=3, percent=100, threshold=0))
        # Load mask (assuming mask has the same name as the image)
        mask_name = self.image_names[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)  # convert to grayscale for binary mask
        
        seed = random.randint(0, 2 ** 32)  # Random seed for consistent transformations
        
        if self.image_transform:
            torch.manual_seed(seed)
            image = self.image_transform(image)

        if self.mask_transform:
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)

        return image, mask

# Define the transformations
image_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])

])

mask_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation(15)], p=0.8),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

valmask_transforms = transforms.Compose([
    transforms.ToTensor()
])


def device_avail():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    return device

#Model Definition
class AE(nn.Module):
    def __init__(self, input_dim, dim1, dim2, dim3, dim4, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim1, 4, 2, 1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(True),
            nn.Conv2d(dim1, dim2, 4, 2, 1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(True),
            nn.Conv2d(dim2, dim3, 4, 2, 1),
            nn.BatchNorm2d(dim3),
            nn.ReLU(True),
            nn.Conv2d(dim3, dim4, 5, 1, 0),
            nn.BatchNorm2d(dim4),
            nn.ReLU(True),
            nn.Conv2d(dim4, z_dim, 3, 1, 0),
            nn.BatchNorm2d(z_dim)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim4, 3, 1, 0),
            nn.BatchNorm2d(dim4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim4, dim3, 5, 1, 0),
            nn.BatchNorm2d(dim3),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim3, dim2, 4, 2, 1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim2, dim1, 4, 2, 1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim1, input_dim, 4, 2, 1),
            #nn.Tanh()
        )


    def forward(self, x):
        z = self.encoder(x)
        x_tilde = self.decoder(z)
        return x_tilde

    
def dice_loss(preds, targets, smooth=1e-6):
    preds= torch.sigmoid(preds)
    threshold = 0.5
    bin_preds = (preds > threshold)
    intersection = (bin_preds * targets).sum()
    union = bin_preds.sum() + targets.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        return bce_loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1):
        super(WeightedDiceLoss, self).__init__()
        self.weights = weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        threshold = 0.5

        binary_mask = (inputs > threshold)

        intersection = (binary_mask * targets).sum()

        dice_pos = (2. * intersection) / (binary_mask.sum() + targets.sum() + self.smooth)

        binary_mask2 = ~binary_mask
        targets2 = 1 - targets

        intersection2 = (binary_mask2 * targets2).sum()

        dice_neg = (2. * intersection2) / (binary_mask2.sum() + targets2.sum() + self.smooth)

       # weighted_dice = self.weights[1] * dice_pos + self.weights[0] * dice_neg
        weighted_dice = dice_pos + dice_neg
        return 1 - weighted_dice


class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, smooth=1, weight=None):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.dice_loss = WeightedDiceLoss(smooth=smooth, weight=weight)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    # Function to compute centroids from binary mask
    def compute_centroids(self, mask):
        # Label the connected components (blobs)
        labeled_mask, num_blobs = label(mask.detach().cpu().numpy())  # Detach from computation graph and convert to NumPy
        centroids = center_of_mass(mask.detach().cpu().numpy(), labeled_mask, range(1, num_blobs + 1))
        centroids = [centroid[:2] for centroid in centroids]  # Keep only X, Y
        return np.array(centroids)  # List of (x, y) coordinates for each centroid

    
    # Function to compute centroid loss
    def centroid_loss(self, pred_mask, true_mask):
        # Get centroids from both masks
        pred_mask = pred_mask.float()
        true_mask = true_mask.float()
        true_centroids = self.compute_centroids(true_mask)
        pred_centroids = self.compute_centroids(pred_mask)
    
        
        if len(true_centroids) == 0 and len(pred_centroids) > 0:
            # No blobs found in one of the masks, return a large penalty
            loss = len(pred_centroids) - len(true_centroids)
            return torch.tensor(loss, dtype=torch.float32, requires_grad=True)
    
        # Match each true centroid with the nearest predicted centroid
        centroid_distances = []
        for true_c in true_centroids:
            distances = np.linalg.norm(pred_centroids - true_c, axis=1)
            min_distance = np.min(distances)
            centroid_distances.append(min_distance)
        
        # Return average centroid distance
        return torch.tensor(np.mean(centroid_distances)/len(true_centroids), requires_grad=True)


    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        cc_loss = self.centroid_loss(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def he_initialization(layer):
    if isinstance(layer, nn.Conv2d):
        # For convolutional layers
        init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Linear):
        # For fully connected layers
        init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            init.constant_(layer.bias, 0)

def train():
    torch.manual_seed(0)
    
    net = AE(1,64,128,256,512,1)  
    device = device_avail()
    net = net.to(device)
    net.apply(he_initialization)

    Optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay= 1e-5)

    low_loss = 1000000000
    # Create the dataset

    tr_dataset = CustomData(
        image_dir='/Train/Images',
        mask_dir='/Train/Labels',
        image_transform=image_transforms,
        mask_transform=mask_transforms
    )

    val_dataset = CustomData(
        image_dir='/Validation/Images',
        mask_dir='/Validation/Labels',
        image_transform=val_transforms,
        mask_transform=valmask_transforms
    )
    
    trdataloader = DataLoader(tr_dataset, batch_size = 16, shuffle=True, num_workers=3)
    vdataloader = DataLoader(val_dataset, batch_size = 16, shuffle=False)
    tr_num_batches = len(trdataloader)
    v_num_batches = len(vdataloader)

    num_pos = 0
    num_neg = 0

    print("calculating class weights")
    for _, masks_batch in trdataloader:
        num_pos += torch.sum(masks_batch)
        num_neg += torch.numel(masks_batch) - torch.sum(masks_batch)

    pos_weight = num_neg / num_pos
    total_pixels = num_neg + num_pos

    class_counts = [num_neg, num_pos]
    class_weights = [1 - (count / total_pixels) for count in class_counts]
    print(class_weights)

    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight,  weight=class_weights)
    criterion2 = WeightedDiceLoss(weight=class_weights)
    criterion3 = nn.BCEWithLogitsLoss()


    num_epochs = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        val_loss = 0.0
        val_loss2 = 0.0
        net.train()  # Set model to training mode
        epoch_loss = 0.0
        running_loss = 0.0

        for x,y in trdataloader:
            x = x.to(device)
            y = y.to(device)

            Optimizer.zero_grad()

            outputs = net(x)

            loss = criterion3(outputs, y)

            running_loss += loss.item()

            loss.backward()

            Optimizer.step()

        epoch_loss = running_loss  / tr_num_batches
        print('{} Epoch Loss: {:.4f} '.format('loss', epoch_loss))

        with torch.no_grad():
            net.eval()
            for real_a, real_b in vdataloader:
                real_a = real_a.to(device)
                real_b = real_b.to(device)

                fake_b = net(real_a)
                loss = criterion3(fake_b, real_b)
                loss2 = dice_loss(fake_b,real_b)
                val_loss += loss.item()
                val_loss2 += loss2.item()

            avg_loss = val_loss / v_num_batches
            avg_loss2 = val_loss2 / v_num_batches
            if  avg_loss < low_loss:
                print('The best loss of generator is', avg_loss)
                low_loss = avg_loss

                best_model_state_dict = copy.deepcopy(net.state_dict())
            print('The validation loss is', avg_loss)
            print("Unweighted DICE score is", avg_loss2)

    net.load_state_dict(best_model_state_dict)
    return net

if __name__ == "__main__":
    print('AE')
    torch.cuda.empty_cache()
    tr_model = train()
    torch.save(tr_model, ' /AE.pth')
