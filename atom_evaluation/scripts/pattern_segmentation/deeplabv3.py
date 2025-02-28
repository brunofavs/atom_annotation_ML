#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from torchvision import models

########## ------- ##########
########## Dataset ##########
########## ------- ##########

class CheckerDataset(Dataset):
    def __init__(self, root_path, limit=None):
        self.root_path = root_path
        self.limit = limit
        self.images = sorted([root_path + "/sample_images/" + i for i in os.listdir(root_path + "/sample_images/")])[:self.limit]
        self.masks = sorted([root_path + "/sample_masks/" + i for i in os.listdir(root_path + "/sample_masks/")])[:self.limit]

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
        
        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return min(len(self.images), self.limit)
    

########## ------- ##########
########## Metrics ##########
########## ------- ##########

def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice


########## -------------- ##########
########## Training tools ##########
########## -------------- ##########

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        """
        Args:
            patience (int): Number of epochs to wait before stopping after no improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter when improvement occurs
        else:
            self.counter += 1  # Increment counter when no improvement

        return self.counter >= self.patience  # Stop training if counter exceeds patience

########## ------------- ##########
########## Testing tools ##########
########## ------------- ##########

def dataset_tester(train_dataset, img_number=1):
    image, mask = train_dataset[img_number]  # Get the first sample
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.permute(1, 2, 0))  # Convert from Tensor (C, H, W) to (H, W, C)
    ax[0].set_title("Image")
    ax[1].imshow(mask.squeeze(), cmap="gray")  # Squeeze single-channel mask
    ax[1].set_title("Mask")
    plt.show()
    exit()

def test_model(model, device,  weights_path="my_checkpoint.pth", image_path = "./random_images/rgbd_hand_color_016.jpg"):
    model.load_state_dict(torch.load(weights_path, weights_only=False))
    num_classes = 1
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    transformation = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
    with torch.no_grad():
        img = Image.open(image_path).convert("RGB")
        shape = img.size
        img_transformed = transformation(img).float().unsqueeze(0).to(device)
        pred = model(img_transformed)
        pred = pred.squeeze(0).permute(1,2,0).cpu()
        pred = pred.squeeze()
        pred[pred < 0] = 0
        pred[pred > 0] = 255
        pred = np.array(pred, dtype=np.uint8)
        pred = cv2.resize(pred, shape)
        # cv2.imshow('teste', pred)
        # cv2.waitKey(0)
        cv2.imwrite("./random_saves/"+"pattern_58.jpg", pred)
        # img_transformed = img_transformed.cpu()
        # plt.figure(figsize=(15, 16))
        # plt.subplot(131), plt.imshow(img_transformed.cpu().detach().squeeze().permute(1, 2, 0)), plt.title("original")
        # plt.subplot(132), plt.imshow(pred, cmap="gray"), plt.title("predicted")
        # plt.show()
    exit()


########## --------- ##########
########## Execution ##########
########## --------- ##########

def main():
    try:
        import rospkg
        rospack = rospkg.RosPack()
        atom_calibration_path = rospack.get_path('atom_calibration')
        atom_evaluation_path = rospack.get_path('atom_evaluation')
        atom_base_path = os.path.commonpath([atom_calibration_path,atom_evaluation_path])
    except:
        atom_base_path = "../../../"

    train_dataset = CheckerDataset(atom_base_path+"/dataset")
    # dataset_tester(train_dataset, 11)

    generator = torch.Generator().manual_seed(25)
    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        num_workers = torch.cuda.device_count() * 4
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 19

    train_dataloader = DataLoader(dataset=train_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    model.to(device)

    # Freeze the backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # test_model(model, device, weights_path="./new_my_checkpoint.pth", image_path = "./random_images/pattern_58.jpg")


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    torch.cuda.empty_cache()
    
    EPOCHS = 200
    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

    # early_stopper = EarlyStopping(patience=5, min_delta=1e-4)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_dc = 0
        
        for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            
            y_pred = model(img)
            print(y_pred)
            optimizer.zero_grad()
            
            dc = dice_coefficient(y_pred, mask)
            loss = criterion(y_pred, mask)
            
            train_running_loss += loss.item()
            train_running_dc += dc.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)
        
        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        model.eval()
        val_running_loss = 0
        val_running_dc = 0
        
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                dc = dice_coefficient(y_pred, mask)
                
                val_running_loss += loss.item()
                val_running_dc += dc.item()

            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)
        
        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
        print("-" * 30)

        # if early_stopper(val_loss):
        #     print(f"Early stopping triggered at epoch {epoch + 1}")
        #     break

    # Saving the model
    torch.save(model.state_dict(), 'my_checkpoint.pth')
    epochs_list = list(range(1, len(train_losses) + 1))
    train_info = {"Epochs": epochs_list, "Train losses": train_losses, "Validation losses": val_losses, "Train DICE": train_dcs, "Validation DICE": val_dcs}
    frame_train_info = pd.DataFrame.from_dict(train_info).set_index('Epochs')
    frame_train_info.to_csv('train_info.csv')
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_list, train_losses, label='Training Loss')
    # plt.plot(epochs_list, val_losses, label='Validation Loss')
    # plt.xticks(ticks=list(range(1, EPOCHS + 1, 1))) 
    # plt.title('Loss over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.tight_layout()

    # plt.legend()


    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_list, train_dcs, label='Training DICE')
    # plt.plot(epochs_list, val_dcs, label='Validation DICE')
    # plt.xticks(ticks=list(range(1, EPOCHS + 1, 1)))  
    # plt.title('DICE Coefficient over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('DICE')
    # plt.grid()
    # plt.legend()

    # plt.tight_layout()
    # plt.savefig("Plot.pdf")
    # plt.show()


if __name__=="__main__":
    main()