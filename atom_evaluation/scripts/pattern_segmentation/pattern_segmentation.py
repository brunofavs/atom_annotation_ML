#!/usr/bin/env python3

import numpy as np
import math

import torch
import torchvision
from torchvision import models
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CalibrationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Convert mask to tensor (values 0 and 1)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, mask


def main():

    # ----------------
    # Model Initialization
    # ----------------

    # Load DeepLabV3 with a ResNet backbone, pretrained on COCO
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)

    # print(model.classifier)
    # exit()

    # Modify the classifier to output 2 classes
    num_classes = 2
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for param in model.backbone.parameters():
        param.requires_grad = False

    # ----------------
    # Dataset Loading
    # ----------------

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = CalibrationDataset(train_image_paths, train_mask_paths, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = CalibrationDataset(test_image_paths, test_mask_paths, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)


    # ---------
    # Training
    # ---------

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']  # Get the segmentation output
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")



    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)['out']
            predictions = torch.argmax(outputs, dim=1)  # Get class labels per pixel



if __name__=="__main__":
    main()
