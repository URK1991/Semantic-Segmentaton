# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import custom_dataset, get_transforms
from model import U_Net
from losses import CombinedBCEDiceLoss, dice_loss

def train_model():
    # Initialize dataset and dataloaders
    train_img_dir = './path_to_train_images'
    train_mask_dir = './path_to_train_masks'
    val_img_dir = './path_to_val_images'
    val_mask_dir = './path_to_val_masks'

    image_transforms, mask_transforms, val_image_transforms, val_mask_transforms = get_transforms()

    train_dataset = custom_dataset(train_img_dir, train_mask_dir, image_transform=image_transforms, mask_transform=mask_transforms)
    val_dataset = custom_dataset(val_img_dir, val_mask_dir, image_transform=val_image_transforms, mask_transform=val_mask_transforms)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer
    model = U_Net(img_ch=1, output_ch=1).cuda()
    criterion = CombinedBCEDiceLoss(bce_weight=1, dice_weight=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
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

            loss = criterion(outputs, y)

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
                loss = criterion(fake_b, real_b)
                loss2 = dice_loss(fake_b,real_b)
                val_loss += loss.item()
                val_loss2 += loss2.item()

            avg_loss = val_loss / v_num_batches
            avg_loss2 = val_loss2 / v_num_batches
            if  avg_loss < low_loss:
                print('The best loss of generator is', avg_loss)
                low_loss = avg_loss

                best_model_state_dict = copy.deepcopy(net.state_dict())
              #  best_model_wts = copy.deepcopy(net.state_dict())
            print('The validation loss is', avg_loss)
            print("Unweighted DICE score is", avg_loss2)

    net.load_state_dict(best_model_state_dict)
    return net

if __name__ == "__main__":
    train_model()
