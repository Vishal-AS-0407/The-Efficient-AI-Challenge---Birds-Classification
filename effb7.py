import os
import time
import json
import pandas as pd
import numpy as np
import torch
import timm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

class BirdDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform=None):
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

data_path = r"/home/effi"
images_path = r"/home/effi/images"
annotations_file = os.path.join(data_path, 'annotation2.csv')
#test_file = os.path.join(data_path, 'test.csv')

annotations = pd.read_csv(annotations_file)

#train_df, test_df = train_test_split(annotations, test_size=0.20, random_state=42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = BirdDataset(annotations, images_path, transform)
#test_dataset = BirdDataset(test_df.sample(n=20), images_path, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('efficientnet_b7', pretrained=False)  # Use MobileNetV3
num_classes = 200
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

def train_model(model, train_loader, loss_fn, optimizer, num_epochs):
    model.train()
    best_accuracy = 0.0
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_time = time.time() - start_time
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, '
              f'Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f}s')
        
        scheduler.step(total_loss / len(train_loader))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')
        
        if epoch==(num_epochs):
            checkpoint_path = os.path.join(checkpoint_dir, f'last_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')

num_epochs = 100
train_model(model, train_loader, loss_fn, optimizer, num_epochs)

