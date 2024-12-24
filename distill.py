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

def main():
    data_path = r"E:\PROJECT\anokha\effi2"
    images_path = r"E:\PROJECT\anokha\effi2\images"
    annotations_file = os.path.join(data_path, 'annotation2.csv')

    annotations = pd.read_csv(annotations_file)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BirdDataset(annotations, images_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=2)

    PRETRAINED = r"E:\PROJECT\anokha\effi2\checkpoints\last_model_epoch_79.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Teacher model (EfficientNet-B7)
    teacher_model = timm.create_model('efficientnet_b7', pretrained=False)
    num_classes = 200
    in_features = teacher_model.classifier.in_features
    teacher_model.classifier = nn.Linear(in_features, num_classes)
    teacher_model.load_state_dict(torch.load(PRETRAINED)["model_state_dict"])
    teacher_model.to(device)
    print("Teacher model loaded successfully")

    # Student model (EdgeNeXt xx-small)
    student_model = timm.create_model('edgenext_xx_small', pretrained=False)
    in_features = student_model.get_classifier().in_features
    student_model.reset_classifier(num_classes=num_classes)
    student_model.to(device)
    print("Student model ready")

    # Distillation loss
    class DistillationLoss(nn.Module):
        def __init__(self, temperature):
            super(DistillationLoss, self).__init__()
            self.temperature = temperature
            self.kl_div = nn.KLDivLoss(reduction='batchmean')

        def forward(self, student_logits, teacher_logits, targets):
            soft_targets = torch.softmax(teacher_logits / self.temperature, dim=1)
            soft_preds = torch.log_softmax(student_logits / self.temperature, dim=1)
            distillation_loss = self.kl_div(soft_preds, soft_targets) * (self.temperature ** 2)
            classification_loss = nn.CrossEntropyLoss()(student_logits, targets)
            return distillation_loss + classification_loss

    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    temperature = 3.0
    criterion = DistillationLoss(temperature)

    teacher_model.eval()
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    # Train the student model
    num_epochs = 100
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in tqdm(train_loader, total=len(train_loader), desc=f'Training {epoch + 1}th epoch'):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            with torch.amp.autocast('cuda'):
                student_logits = student_model(images)
                loss = criterion(student_logits, teacher_logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum().item()

        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
              f'Accuracy: {accuracy:.4f}')

        scheduler.step(running_loss / len(train_loader))
        torch.save(student_model.state_dict(), f'student_model{epoch}.pth')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        torch.save(student_model.state_dict(), f'E:/PROJECT/anokha/effi2/models/student_model{epoch}.pth')

    print('Finished Training')
    # Save the trained student model

if __name__ == "__main__":
    main()
