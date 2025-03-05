#!/usr/bin/env python3

import copy
import os
import random
import shutil
import zipfile
from math import atan2, cos, sin, sqrt, pi, log

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
import os
import pandas as pd


import rospkg


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out
    
class CarvanaDataset(Dataset):
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
    

def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice


def test_model(model, device,  weights_path="my_checkpoint.pth", image_path = "./random_images/rgbd_hand_color_016.jpg"):
    model.load_state_dict(torch.load(weights_path, weights_only=False))

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
        cv2.imwrite("./random_saves/"+"rgbd_hand_color_190.jpg", pred)
        # img_transformed = img_transformed.cpu()
        # plt.figure(figsize=(15, 16))
        # plt.subplot(131), plt.imshow(img_transformed.cpu().detach().squeeze().permute(1, 2, 0)), plt.title("original")
        # plt.subplot(132), plt.imshow(pred, cmap="gray"), plt.title("predicted")
        # plt.show()
    exit()


# input_image = torch.rand((1,3,512,512))
# model = UNet(3,10)
# output = model(input_image)
# print(output.size())
# You should get torch.Size([1, 10, 512, 512]) as a result
def main():
    rospack = rospkg.RosPack()
    atom_calibration_path = rospack.get_path('atom_calibration')
    atom_evaluation_path = rospack.get_path('atom_evaluation')
    atom_base_path = os.path.commonpath([atom_calibration_path,atom_evaluation_path])
    train_dataset = CarvanaDataset(atom_base_path+"/dataset")

    # image, mask = train_dataset[11]  # Get the first sample
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(image.permute(1, 2, 0))  # Convert from Tensor (C, H, W) to (H, W, C)
    # ax[0].set_title("Image")
    # ax[1].imshow(mask.squeeze(), cmap="gray")  # Squeeze single-channel mask
    # ax[1].set_title("Mask")
    # plt.show()
    # exit()

    generator = torch.Generator().manual_seed(25)
    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        num_workers = torch.cuda.device_count() * 4
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 20

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

    model = UNet(in_channels=3, num_classes=1).to(device)
    # test_model(model, device, weights_path="my_checkpoint200.pth", image_path = "./random_images/rgbd_hand_color_190.jpg")

    # model.load_state_dict(torch.load("my_checkpoint200.pth", weights_only=False))
    
    # num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(num_params)
    # exit()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    torch.cuda.empty_cache()
    
    EPOCHS = 200
    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_dc = 0
        
        for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            
            y_pred = model(img)
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

    # Saving the model
    torch.save(model.state_dict(), 'my_checkpoint.pth')
    epochs_list = list(range(1, len(train_losses) + 1))
    train_info = {"Epochs": epochs_list, "Train losses": train_losses, "Validation losses": val_losses, "Train DICE": train_dcs, "Validation DICE": val_dcs}
    frame_train_info = pd.DataFrame.from_dict(train_info).set_index('Epochs')
    frame_train_info.to_csv('train_info.csv')

if __name__=="__main__":
    main()