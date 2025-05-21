# Import necessary libraries
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

# Set the dataset path and create the directory if it doesn't exist
dataset_path = "./chest_xray_pneumonia"
os.makedirs(dataset_path, exist_ok=True)

# Download the dataset using Kaggle CLI if it is not already present
if not os.path.exists(os.path.join(dataset_path, "chest_xray")):
    os.system(f"kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p {dataset_path} --unzip")
else:
    print("Dataset already exists. Skipping download.")

# Define training and testing directory paths
train_dir = os.path.join(dataset_path, "chest_xray", "train")
test_dir = os.path.join(dataset_path, "chest_xray", "test")

# Ensure that the dataset has been successfully downloaded and extracted
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("Dataset not found. Please check the download path.")

# Load all image paths and labels from the training dataset
categories = ["NORMAL", "PNEUMONIA"]
data = []
labels = []

for category in categories:
    path = os.path.join(train_dir, category)
    for img_path in glob.glob(os.path.join(path, "*.jpeg")):
        data.append(img_path)
        labels.append(category)

# Create a DataFrame from image paths and their corresponding labels
df = pd.DataFrame({"image_path": data, "label": labels})

# Convert labels to binary: NORMAL -> 0, PNEUMONIA -> 1
lb = LabelBinarizer()
y = lb.fit_transform(df["label"]).flatten()

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df["image_path"], y, test_size=0.2, random_state=42, stratify=y)

# Define image preprocessing transformations
data_transform = transforms.Compose([
    transforms.Resize((150, 150)),       # Resize all images to 150x150
    transforms.ToTensor(),               # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
])

# Custom PyTorch dataset class for chest X-ray images
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths.values
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = datasets.folder.default_loader(self.image_paths[idx])  # Load image
        if self.transform:
            img = self.transform(img)  # Apply transformations
        return img, self.labels[idx]

# Create PyTorch datasets and data loaders
dataset_train = ChestXRayDataset(X_train, y_train, transform=data_transform)
dataset_test = ChestXRayDataset(X_test, y_test, transform=data_transform)
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

# Define a Convolutional Neural Network (CNN) for binary classification
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# Handle class imbalance by weighting the loss function
pos_weight = torch.tensor([sum(y_train == 0) / sum(y_train == 1)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop for 10 epochs
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

# Evaluate model performance on the test set
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
        outputs = torch.sigmoid(model(batch_X))  # Apply sigmoid to get probabilities
        y_pred.append((outputs > 0.5).cpu().numpy())  # Convert to binary predictions
        y_true.append(batch_y.cpu().numpy())

# Stack predictions and true labels
y_pred = np.vstack(y_pred)
y_true = np.vstack(y_true)

# Compute evaluation metrics
roc_score = roc_auc_score(y_true, y_pred)
print("CNN Accuracy:", accuracy_score(y_true, y_pred))
print("ROC AUC Score:", roc_score)
print(classification_report(y_true, y_pred, target_names=lb.classes_))

