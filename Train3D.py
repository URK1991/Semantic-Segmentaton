import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from PostProcessing.utils import reshape_img, normalize
import pandas as pd
from 3Dmodels import *
from torch.autograd import Variable
import copy
from skimage.filters import unsharp_mask
from skimage.filters import threshold_otsu
import random
from scipy.ndimage import rotate, zoom, shift
from skimage import transform
from torch.nn.utils import clip_grad_norm_

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        N = inputs.size()[0]
        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(N, -1)
        targets = targets.contiguous().view(N, -1)
        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)

        return 1 - dice.sum() / N


class Custom(Dataset):
    def __init__(self, data_dir, label_dir, img_size, transform=None, is_normal=True):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.image_names = os.listdir(data_dir)
        self.transform = transform
        self.output_size = img_size
        self.is_normal = is_normal

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_dir, self.image_names[idx])
        mask_name = self.image_names[idx]
        mask_path = os.path.join(self.label_dir, mask_name)
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()
        label_nii = nib.load(mask_path)
        label = label_nii.get_fdata()
        img = reshape_img(img, self.output_size)
        label = reshape_img(label, self.output_size)
        img = np.expand_dims(img, 0)
        label = np.expand_dims(label, 0)
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        sample = {'image': img, 'label': label}

        return sample


def device_avail():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    return device

def train():
    device = device_avail()

    criterion1 = torch.nn.MSELoss()
    criterion2 = DiceLoss()

    # Loss weight of L1 voxel-wise loss between translated image and real image
    lambda_voxel = 100
    batch_size = 1
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, 384// 2 ** 4, 384 // 2 ** 4, 256 // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


    # Configure dataloaders
    tr_dataset = Custom(
        data_dir=-'.../Train/Images',
        label_dir='.../Train/Labels',
        img_size = [384,384,256] #you can set the dimensions based on the available computational and memory resources
    )

    v_dataset = Custom(
        data_dir='.../Validation/Images',
        label_dir='.../Validation/Labels',
        img_size = [384,384,256] #The same as the training set
    )

    dataloader = DataLoader(tr_dataset, batch_size, num_workers=6, shuffle=True)
    vdataloader = DataLoader(v_dataset, batch_size, num_workers=6, shuffle=False)
    # Training Loop
    prev_loss = 10000
    low_loss = prev_loss
    Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
    num_epochs = 50
    valid = torch.ones((batch_size, *patch), dtype=torch.float32, device=device, requires_grad=False)
    fake = torch.zeros((batch_size, *patch), dtype=torch.float32, device=device, requires_grad=False)

    for epoch in range(num_epochs):
        total_loss_G = 0
        total_loss_D = 0
        loss_v = 0
        generator.train()
        discriminator.train()
        for i, batch in enumerate(dataloader):
            # Model inputs
            real_A = batch["image"].to(device).float()
            real_B = batch["label"].to(device).float()

            fake_B = generator(real_A)

            pred_real = discriminator(real_B, real_A)
            loss_real = criterion1(pred_real, valid)

            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion1(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            optimizer_D.zero_grad()
            loss_D.backward()

            #Performing gradient clipping to prevent gradient exploding
            clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # 1.0 is a typical value for max_norm. 

            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            del loss_real, loss_fake, pred_real, pred_fake

            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion1(pred_fake, valid)
            loss_voxel = criterion2(fake_B, real_B)
            # Total loss
            
          loss_G = loss_GAN + lambda_voxel * loss_voxel

            loss_G.backward()

            #performing gradient clipping 
            clip_grad_norm_(generator.parameters(), max_norm=1.0)  # 1.0 is a typical value for max_norm

            optimizer_G.step()

            del real_A, real_B, fake_B, pred_fake #freeing memory resources
      
            torch.cuda.empty_cache()

            total_loss_D += loss_GAN
            total_loss_G += loss_voxel

        mean_loss_G = total_loss_G / len(dataloader)
        mean_loss_D = total_loss_D / len(dataloader)

        print('epoch number', epoch)
        print('GAN loss', mean_loss_D)
        print('voxel loss', mean_loss_G)

        with torch.no_grad():
            generator.eval()
            for i, batch in enumerate(vdataloader):
                a = batch["image"].to(device).float()
                b = batch["label"].to(device).float()

                f_b = generator(a)
                f_b = (f_b > 0.5).float()
                loss_voxel_v += criterion2(f_b, b)

        loss_v = loss_v/len(vdataloader)
        print('pixel loss for validation data', loss_v)

        if loss_v < prev_loss:
            low_loss = loss_v
            best_model_state_dict = copy.deepcopy(generator.state_dict())
            prev_loss = low_loss

        del a, b, f_b
        torch.cuda.empty_cache()

    generator.load_state_dict(best_model_state_dict)
    return generator

if __name__ == "__main__":
    print('training a 3D segmentation model')
    torch.cuda.empty_cache( )
    tr_model = train()
    torch.save(tr_model, '../model.pth')
