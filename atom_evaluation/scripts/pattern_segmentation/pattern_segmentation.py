#!/usr/bin/env python3

import numpy as np
import os
import math

import rospkg

import torch
import torchvision
from torchvision import models
import torch.nn as nn
from torchinfo import summary

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CalibrationDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_names = os.listdir(images_path)
        self.masks_names  = os.listdir(masks_path)
        self.transform    = transform

        self.images_path  = images_path
        self.masks_path   = masks_path

        assert len(os.listdir(images_path)) == len(os.listdir(masks_path))
        # print(f'{self.images_path}/{self.images_names[0]}')
        
        first_image = Image.open(os.path.join(self.images_path,self.images_names[0]))
        first_mask = Image.open(os.path.join(self.masks_path,self.masks_names[0]))

        # first_image.show()
        # first_mask.show()

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.images_path,self.images_names[idx]))
        mask = Image.open(os.path.join(self.masks_path,self.masks_names[idx]))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def main():

    # from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
    # weights = DeepLabV3_ResNet50_Weights.DEFAULT
    # category_name = weights.meta["categories"]
    # print(category_name)
    # exit()

    # ----------------
    # Find ATOM base path
    # ----------------

    rospack = rospkg.RosPack()
    atom_calibration_path = rospack.get_path('atom_calibration')
    atom_evaluation_path = rospack.get_path('atom_evaluation')
    atom_base_path = os.path.commonpath([atom_calibration_path,atom_evaluation_path])


    # ----------------
    # Model Initialization
    # ----------------

    # Load DeepLabV3 with a ResNet backbone, pretrained on COCO
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')

    # print(model)
    # exit()

    # print(model.classifier)
    # exit()

    # Modify the classifier to output 2 classes
    num_classes = 2
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    # print(model)
    # exit()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # summary(model, input_size=(4, 3, 256, 256))

    for param in model.backbone.parameters():
        param.requires_grad = False

    # summary(model, input_size=(4, 3, 256, 256))
    # exit()


    # ----------------
    # Dataset Loading
    # ----------------

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_images_path = f'{atom_base_path}/segmentation_dataset/train/images'
    train_masks_path = f'{atom_base_path}/segmentation_dataset/train/masks'

    test_images_path = f'{atom_base_path}/segmentation_dataset/test/images'
    test_masks_path = f'{atom_base_path}/segmentation_dataset/test/masks'

    train_dataset = CalibrationDataset(train_images_path, train_masks_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = CalibrationDataset(test_images_path, test_masks_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # ---------
    # Training
    # ---------

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        print(f'Starting epoch {epoch}')
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            masks = masks.squeeze(1) # Shape: [4, 1, 256, 256] to [4, 256, 256]

            
            optimizer.zero_grad()
            outputs = model(images)['out']  # Get the segmentation output


            predictions = torch.argmax(outputs, dim=1)  # Shape: [4, 256, 256]
            predictions = predictions.float()  # Ensure logits are float

            # print(predictions.dtype)
            # print(masks.dtype)

            loss = criterion(predictions, masks)
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")



    # model.eval()
    # with torch.no_grad():
    #     for images, _ in test_loader:
    #         images = images.to(device)
    #         outputs = model(images)['out']
    #         predictions = torch.argmax(outputs, dim=1)  # Get class labels per pixel
    #         print(predictions)


if __name__=="__main__":
    main()
