import os
import torch
import timm  # Ensure timm is imported
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm


# Define the BirdDataset class
class BirdDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]  # Get image path from CSV
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# Load test.csv and initialize necessary paths
data_path = r"/home/effi"
test_file = os.path.join(data_path, 'test.csv')

# Load test file with image paths
test_df = pd.read_csv(test_file)

# Set up the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set up the test dataset and dataloader
test_dataset = BirdDataset(test_df, transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load the model architecture and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('efficientnet_b7', pretrained=False)  # Create the model architecture
model.classifier = torch.nn.Linear(model.classifier.in_features, 200)  # Adjust for your number of classes
checkpoint = torch.load(r'/home/effi/checkpoints/best_model_epoch_79.pth')  # Load the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])  # Load only the model state dict
model.to(device)
model.eval()

# Prediction function
def test_model(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images in tqdm(test_loader,total=len(test_loader),desc="generating submission file"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions

# Run predictions
predictions = test_model(model, test_loader)

# Add predictions to test_df and save the updated CSV file
test_df['class'] = predictions
test_df.to_csv(os.path.join(data_path, 'submission.csv'), index=False)

print("Predictions appended and saved to submission.csv")
